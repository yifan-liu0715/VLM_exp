#!/usr/bin/env python3
"""Run Scheme 4 at the exact native-resolution Scheme 3 conditions."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
from peft import LoraConfig
import torch
from transformers import AutoTokenizer

from infer_scheme3_native import (
    load_aligned_sample as load_native_aligned_sample,
)
from infer_scheme4 import (
    atomic_json,
    build_teacher_scheduler,
    encode_prompt,
    load_manifest_row,
    load_teacher_adapter,
    make_comparison,
    make_seeded_noise,
    run_denoising,
    save_mp4,
    sha256_file,
    tensor_to_uint8_frames,
)
from install_reference_position_encoding import (
    install_reference_position_encoding,
)
from native_reference_position import (
    interpolate_reference_position_for_native_grid,
)
from teacher_aligned_system import (
    ConditionAttnProcessor2_0,
    WanTransformer3DModel,
    load_text_encoder,
    load_vae,
)
from training_contracts import explicit_nonzero_visible_gpus


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCHEME3_ROOT = EXPERIMENT_ROOT.parent / "v2v_native_resolution_scheme3_0727"
DEFAULT_NATIVE_CONFIG = SCHEME3_ROOT / "configs/train_scheme3_native_2000.yaml"
DEFAULT_SCHEME4_CONFIG = EXPERIMENT_ROOT / "configs/train_scheme4_gate_1000.yaml"
DEFAULT_ADAPTER = (
    EXPERIMENT_ROOT
    / "recommended/final_adapter/v2v_lora_source_reference_posenc.safetensors"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT / "validation/native_supplement_scheme3_conditions_0730"
)

TRAINED_CANVAS_BY_NATIVE = {
    (512, 1280): (384, 960),
    (704, 1280): (352, 640),
    (1280, 704): (640, 352),
}


def assert_scheme3_conditions(native_cfg, scheme4_cfg) -> None:
    expected = {
        "num_frames": 97,
        "steps": 50,
        "guidance_scale": 5.0,
        "flow_shift": 5.0,
        "seed": 42,
    }
    actual = {
        "num_frames": int(native_cfg.data.num_frames),
        "steps": int(native_cfg.inference.steps),
        "guidance_scale": float(native_cfg.inference.guidance_scale),
        "flow_shift": float(native_cfg.inference.flow_shift),
        "seed": int(native_cfg.inference.seed),
    }
    if actual != expected:
        raise ValueError(
            f"native Scheme 3 inference contract mismatch: {actual}"
        )
    for key in ("steps", "guidance_scale", "flow_shift", "seed"):
        if actual[key] != {
            "steps": int(scheme4_cfg.inference.steps),
            "guidance_scale": float(scheme4_cfg.inference.guidance_scale),
            "flow_shift": float(scheme4_cfg.inference.flow_shift),
            "seed": int(scheme4_cfg.inference.seed),
        }[key]:
            raise ValueError(f"Scheme 3/Scheme 4 mismatch for {key}")


def patch_grid(
    transformer: torch.nn.Module,
    latent: torch.Tensor,
) -> tuple[int, int]:
    _, patch_height, patch_width = transformer.config.patch_size
    return (
        int(latent.shape[-2]) // int(patch_height),
        int(latent.shape[-1]) // int(patch_width),
    )


def trained_grid_for_native_canvas(
    native_canvas: tuple[int, int],
) -> tuple[int, int]:
    if native_canvas not in TRAINED_CANVAS_BY_NATIVE:
        raise ValueError(f"unsupported native canvas: {native_canvas}")
    trained_height, trained_width = TRAINED_CANVAS_BY_NATIVE[native_canvas]
    # Wan VAE spatial stride 8 followed by transformer patch stride 2.
    return trained_height // 16, trained_width // 16


def read_video(path: Path, num_frames: int) -> np.ndarray:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if len(reader) != int(num_frames):
        raise RuntimeError(
            f"{path} contains {len(reader)} frames, expected {num_frames}"
        )
    return reader.get_batch(list(range(int(num_frames)))).asnumpy()


def baseline_paths(sample_id: str) -> dict[str, Path]:
    root = SCHEME3_ROOT / "validation" / sample_id
    videos = root / "videos"
    return {
        "source": videos / f"{sample_id}_Source.mp4",
        "target": videos / f"{sample_id}_Target.mp4",
        "base": videos / f"{sample_id}_Base.mp4",
        "scheme3": videos / f"{sample_id}_NativeResolution_LoRA.mp4",
        "manifest": (
            root / "reports" / f"{sample_id}_run_manifest.json"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--native-config", type=Path, default=DEFAULT_NATIVE_CONFIG)
    parser.add_argument("--scheme4-config", type=Path, default=DEFAULT_SCHEME4_CONFIG)
    parser.add_argument("--scheme4-adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    native_cfg = OmegaConf.load(args.native_config)
    scheme4_cfg = OmegaConf.load(args.scheme4_config)
    assert_scheme3_conditions(native_cfg, scheme4_cfg)
    row = load_manifest_row(native_cfg.data.test_manifest, args.sample_id)
    baselines = baseline_paths(args.sample_id)
    missing = [str(path) for path in baselines.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Scheme 3 baselines are missing: {missing}")
    if not args.scheme4_adapter.is_file():
        raise FileNotFoundError(args.scheme4_adapter)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "sample_id": args.sample_id,
                    "source": row["input_video"],
                    "target": row["output_video"],
                    "reference": row["reference_image"],
                    "scheme3_manifest": str(baselines["manifest"]),
                    "scheme4_adapter": str(args.scheme4_adapter),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    visible = explicit_nonzero_visible_gpus(
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    if len(visible) != 1:
        raise RuntimeError("inference requires exactly one physical nonzero GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(int(native_cfg.inference.seed))
    np.random.seed(int(native_cfg.inference.seed))

    sample = load_native_aligned_sample(
        row,
        num_frames=int(native_cfg.data.num_frames),
        bucket_align=int(native_cfg.data.bucket_align),
    )
    native_canvas = (int(sample["height"]), int(sample["width"]))
    trained_canvas = TRAINED_CANVAS_BY_NATIVE[native_canvas]

    tokenizer = AutoTokenizer.from_pretrained(
        native_cfg.model.tokenizer,
        local_files_only=True,
    )
    text_encoder = (
        load_text_encoder(native_cfg.model.text_encoder)
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    vae = load_vae(native_cfg.model.vae).eval().requires_grad_(False).to(device)
    transformer = WanTransformer3DModel.from_pretrained(
        native_cfg.model.transformer,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=int(scheme4_cfg.lora.rank),
            lora_alpha=int(scheme4_cfg.lora.alpha),
            lora_dropout=float(scheme4_cfg.lora.dropout),
            init_lora_weights=True,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    for block in transformer.blocks:
        block.attn1.set_processor(ConditionAttnProcessor2_0())
    transformer.patch_embedding_extra = copy.deepcopy(
        transformer.patch_embedding
    ).requires_grad_(False)
    install_reference_position_encoding(
        transformer,
        scheme4_cfg,
        trainable=False,
        enabled=True,
    )
    transformer = transformer.eval().to(device)
    scheme4_summary = load_teacher_adapter(
        transformer,
        args.scheme4_adapter,
        require_reference_position=True,
    )

    source_tensor = sample["source"].to(device)
    reference_tensor = sample["reference"].to(device)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        source_latent = vae.model.encode(source_tensor.float(), vae.scale)
        reference_latent = vae.model.encode(reference_tensor.float(), vae.scale)
    native_grid = patch_grid(transformer, reference_latent)
    trained_grid = trained_grid_for_native_canvas(native_canvas)
    interpolation_report = interpolate_reference_position_for_native_grid(
        transformer.reference_position_encoder,
        trained_grid=trained_grid,
        native_grid=native_grid,
    )
    interpolation_report["trained_canvas"] = list(trained_canvas)
    interpolation_report["native_canvas"] = list(native_canvas)
    interpolation_report["inference_only"] = True

    prompt_embeddings = encode_prompt(
        tokenizer,
        text_encoder,
        row["caption"],
        device,
    )
    negative_prompt_embeddings = encode_prompt(
        tokenizer,
        text_encoder,
        "",
        device,
    )
    initial_noise = make_seeded_noise(
        tuple(source_latent.shape),
        int(native_cfg.inference.seed),
        device,
    )
    scheme4_latent = run_denoising(
        transformer,
        build_teacher_scheduler(native_cfg),
        initial_noise,
        prompt_embeddings,
        negative_prompt_embeddings,
        source_latent.to(torch.bfloat16),
        reference_latent.to(torch.bfloat16),
        steps=int(native_cfg.inference.steps),
        guidance_scale=float(native_cfg.inference.guidance_scale),
        label="scheme4-native-supplement",
    )

    del transformer, text_encoder, prompt_embeddings
    del negative_prompt_embeddings, source_latent, reference_latent
    torch.cuda.empty_cache()
    scheme4_video = tensor_to_uint8_frames(
        vae.decode(scheme4_latent, device=device, tiled=True)
    )
    del vae, scheme4_latent
    torch.cuda.empty_cache()

    frame_count = int(native_cfg.data.num_frames)
    baseline_frames = {
        name: read_video(path, frame_count)
        for name, path in baselines.items()
        if name != "manifest"
    }
    expected_shape = (
        frame_count,
        int(sample["height"]),
        int(sample["width"]),
        3,
    )
    for name, frames in {
        **baseline_frames,
        "scheme4": scheme4_video,
    }.items():
        if tuple(frames.shape) != expected_shape:
            raise RuntimeError(
                f"{name} shape {tuple(frames.shape)} != {expected_shape}"
            )

    output_root = args.output_dir / args.sample_id
    video_root = output_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    baseline_names = {
        "source": "Source.mp4",
        "target": "Target.mp4",
        "base": "Base.mp4",
        "scheme3": "Scheme3.mp4",
    }
    for name, output_name in baseline_names.items():
        destination = video_root / output_name
        shutil.copy2(baselines[name], destination)
        output_paths[name] = str(destination)
    scheme4_path = video_root / "Scheme4_NativePEInterpolated.mp4"
    save_mp4(scheme4_video, scheme4_path, float(sample["fps"]))
    output_paths["scheme4"] = str(scheme4_path)

    comparisons = {
        "comparison_base_scheme4": make_comparison(
            baseline_frames["source"],
            baseline_frames["target"],
            baseline_frames["base"],
            scheme4_video,
            lower_left_label="BASE (Scheme 3 native run)",
            lower_right_label="SCHEME 4 (native PE interpolation)",
        ),
        "comparison_scheme3_scheme4": make_comparison(
            baseline_frames["source"],
            baseline_frames["target"],
            baseline_frames["scheme3"],
            scheme4_video,
            lower_left_label="SCHEME 3 (native)",
            lower_right_label="SCHEME 4 (native PE interpolation)",
        ),
    }
    for name, frames in comparisons.items():
        destination = video_root / f"{name}_4grid.mp4"
        save_mp4(frames, destination, float(sample["fps"]))
        output_paths[name] = str(destination)

    baseline_manifest = json.loads(
        baselines["manifest"].read_text(encoding="utf-8")
    )
    report = {
        "sample_id": args.sample_id,
        "split": row["source_split"],
        "source_video": row["input_video"],
        "target_video": row["output_video"],
        "reference_image": row["reference_image"],
        "prompt": row["caption"],
        "height": int(sample["height"]),
        "width": int(sample["width"]),
        "num_frames": int(native_cfg.data.num_frames),
        "fps": float(sample["fps"]),
        "scheduler": "UniPCMultistepScheduler",
        "flow_shift": float(native_cfg.inference.flow_shift),
        "steps": int(native_cfg.inference.steps),
        "guidance_scale": float(native_cfg.inference.guidance_scale),
        "seed": int(native_cfg.inference.seed),
        "physical_gpu": visible[0],
        "scheme3_manifest": str(baselines["manifest"]),
        "scheme3_manifest_sha256": sha256_file(baselines["manifest"]),
        "scheme3_adapter_sha256": baseline_manifest["adapter_sha256"],
        "scheme4_adapter": str(args.scheme4_adapter),
        "scheme4_adapter_sha256": sha256_file(args.scheme4_adapter),
        "scheme4_adapter_summary": scheme4_summary,
        "identical_initial_noise_with_scheme3": True,
        "identical_source_reference_conditions": True,
        "geometry_preprocessing": (
            "native_source_target_reference_aspect_fit_replicate_padding"
        ),
        "reference_position_interpolation": interpolation_report,
        "reused_scheme3_native_baseline": True,
        "outputs": output_paths,
        "quality_metrics_computed": False,
    }
    manifest_path = output_root / "run_manifest.json"
    atomic_json(manifest_path, report)
    print(f"[done] scheme4={scheme4_path}", flush=True)
    print(f"[done] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
