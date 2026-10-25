#!/usr/bin/env python3
"""Run the Scheme4 V4 model stack on the normalized comparison input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path


INFERENCE_VARIANTS = ("scheme4_v4",)
DIFFSYNTH_PROJECT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
SCHEME4_ROOT = DIFFSYNTH_PROJECT / "v2v_reference_independent_posenc_scheme4_0729"
SCHEME4_SCRIPTS = SCHEME4_ROOT / "scripts"
TEACHER_REFERENCE = SCHEME4_ROOT / "teacher_reference"
PYTHON_PACKAGES = DIFFSYNTH_PROJECT / "v2v_formal_0715/environment/python_packages"
TRANSFORMER_PATH = DIFFSYNTH_PROJECT / "v2v_formal_0715/base_components/transformer"
TEXT_ENCODER_PATH = (
    DIFFSYNTH_PROJECT
    / "pretrain_weights/Wan2.2-TI2V-5B-HF/models_t5_umt5-xxl-enc-bf16.pth"
)
VAE_PATH = DIFFSYNTH_PROJECT / "pretrain_weights/Wan2.2-TI2V-5B-HF/Wan2.2_VAE.pth"
TOKENIZER_PATH = DIFFSYNTH_PROJECT / "pretrain_weights/Wan2.2-TI2V-5B-HF/google/umt5-xxl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != str(expected).lower():
        raise ValueError(f"SHA256 mismatch for {path}: {observed} != {expected}")
    return observed


def parse_visible_gpu(value: str | None) -> str:
    normalized = "" if value is None else str(value).strip()
    if not normalized.isdigit() or normalized == "0":
        raise ValueError("CUDA_VISIBLE_DEVICES must contain exactly one physical nonzero GPU")
    return normalized


def initialize_cuda_for_inference() -> None:
    import torch

    torch.cuda.set_device(torch.device("cuda:0"))


def validate_input_manifest(payload: dict, *, require_files: bool = True) -> dict:
    from mode_comparison.contracts import DANCE_SAMPLE_ID, PROMPT

    expected = {
        "sample_id": DANCE_SAMPLE_ID,
        "prompt": PROMPT,
        "width": 512,
        "height": 928,
        "num_frames": 65,
        "fps": 16,
        "target_usage": "evaluation-only; excluded from generation conditioning",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"input manifest {key}={payload.get(key)!r}, expected {value!r}")
    for key in ("source_video", "target_video", "reference_image"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"input manifest is missing {key}")
        if require_files and not Path(payload[key]).is_file():
            raise FileNotFoundError(payload[key])
    expected_conditioning = [
        payload["source_video"],
        payload["reference_image"],
        payload["prompt"],
    ]
    if payload.get("generation_conditioning") != expected_conditioning:
        raise ValueError("generation conditioning must be exactly source, reference, and prompt")
    if payload["target_video"] in payload["generation_conditioning"]:
        raise ValueError("target video cannot be generation conditioning")
    return dict(payload)


def build_runtime_config(payload: dict) -> dict:
    return {
        "model": {
            "transformer": str(TRANSFORMER_PATH),
            "text_encoder": str(TEXT_ENCODER_PATH),
            "vae": str(VAE_PATH),
            "tokenizer": str(TOKENIZER_PATH),
        },
        "data": {
            "num_frames": int(payload["num_frames"]),
            "fps": int(payload["fps"]),
            "canvas_by_bucket": {"512x928": [928, 512]},
        },
        "reference_position": {
            "enabled": True,
            "max_frames": 4,
            "max_height": 64,
            "max_width": 64,
            "init_std": 0.02,
        },
        "lora": {"rank": 96, "alpha": 96, "dropout": 0.0},
        "inference": {
            "steps": 40,
            "guidance_scale": 5.0,
            "flow_shift": 3.0,
            "seed": 42,
        },
    }


def inspect_video(path: Path) -> dict[str, object]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if not len(reader):
        raise ValueError(f"video is empty: {path}")
    frame = reader[0].asnumpy()
    return {
        "frames": len(reader),
        "fps": float(reader.get_avg_fps()),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def preflight(args: argparse.Namespace) -> tuple[dict, dict]:
    physical_gpu = parse_visible_gpu(os.environ.get("CUDA_VISIBLE_DEVICES"))
    payload = validate_input_manifest(
        json.loads(args.input_manifest.read_text(encoding="utf-8"))
    )
    adapter_sha256 = verify_sha256(args.adapter, args.expected_adapter_sha256)
    for path in (TRANSFORMER_PATH, TEXT_ENCODER_PATH, VAE_PATH, TOKENIZER_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_file.exists() or args.run_manifest.exists():
        raise FileExistsError("refusing to overwrite existing LoRA output or run manifest")
    source_probe = inspect_video(Path(payload["source_video"]))
    target_probe = inspect_video(Path(payload["target_video"]))
    expected_probe = {"frames": 65, "fps": 16.0, "width": 512, "height": 928}
    if source_probe != expected_probe or target_probe != expected_probe:
        raise ValueError(
            f"normalized video mismatch: source={source_probe}, target={target_probe}"
        )
    from PIL import Image

    with Image.open(payload["reference_image"]) as reference:
        if reference.size != (512, 928):
            raise ValueError(f"reference size is {reference.size}, expected (512, 928)")
    record = {
        "status": "passed",
        "physical_gpu": physical_gpu,
        "input_manifest": str(args.input_manifest),
        "input_manifest_sha256": sha256_file(args.input_manifest),
        "adapter": str(args.adapter),
        "adapter_sha256": adapter_sha256,
        "source_probe": source_probe,
        "target_probe": target_probe,
        "runtime_config": build_runtime_config(payload),
    }
    return payload, record


def _load_normalized_sample(payload: dict, torch):
    import numpy as np
    from decord import VideoReader, cpu
    from PIL import Image

    indices = list(range(65))
    source = VideoReader(payload["source_video"], ctx=cpu(0), num_threads=1)
    target = VideoReader(payload["target_video"], ctx=cpu(0), num_threads=1)

    def video_tensor(reader):
        array = reader.get_batch(indices).asnumpy()
        return (
            torch.from_numpy(array)
            .to(torch.float32)
            .permute(0, 3, 1, 2)
            .div(127.5)
            .sub(1.0)
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
            .contiguous()
        )

    with Image.open(payload["reference_image"]) as handle:
        reference_array = np.asarray(handle.convert("RGB")).copy()
    reference = (
        torch.from_numpy(reference_array)
        .to(torch.float32)
        .permute(2, 0, 1)
        .div(127.5)
        .sub(1.0)
        .unsqueeze(0)
        .unsqueeze(2)
        .contiguous()
    )
    return {
        "source": video_tensor(source),
        "target": video_tensor(target),
        "reference": reference,
    }


def run_inference(args: argparse.Namespace, payload: dict, preflight_record: dict) -> dict:
    for path in (PYTHON_PACKAGES, SCHEME4_SCRIPTS, TEACHER_REFERENCE, DIFFSYNTH_PROJECT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import copy
    import numpy as np
    from omegaconf import OmegaConf
    from peft import LoraConfig
    import torch
    from transformers import AutoTokenizer

    import infer_scheme4 as original
    from install_reference_position_encoding import install_reference_position_encoding
    from teacher_aligned_system import (
        ConditionAttnProcessor2_0,
        WanTransformer3DModel,
        load_text_encoder,
        load_vae,
    )

    cfg = OmegaConf.create(build_runtime_config(payload))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.inference.seed))
    np.random.seed(int(cfg.inference.seed))
    sample = _load_normalized_sample(payload, torch)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer,
        local_files_only=True,
    )
    text_encoder = (
        load_text_encoder(cfg.model.text_encoder)
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    vae = load_vae(cfg.model.vae).eval().requires_grad_(False).to(device)
    transformer = WanTransformer3DModel.from_pretrained(
        cfg.model.transformer,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=int(cfg.lora.rank),
            lora_alpha=int(cfg.lora.alpha),
            lora_dropout=float(cfg.lora.dropout),
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
        cfg,
        trainable=False,
        enabled=False,
    )
    transformer = transformer.eval().to(device)

    source_tensor = sample["source"].to(device)
    reference_tensor = sample["reference"].to(device)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        source_latent = vae.model.encode(source_tensor.float(), vae.scale)
        reference_latent = vae.model.encode(reference_tensor.float(), vae.scale)
    prompt_embeddings = original.encode_prompt(
        tokenizer, text_encoder, payload["prompt"], device
    )
    negative_prompt_embeddings = original.encode_prompt(
        tokenizer, text_encoder, "", device
    )
    initial_noise = original.make_seeded_noise(
        tuple(source_latent.shape), int(cfg.inference.seed), device
    )
    transformer.reference_position_encoding_enabled = True
    adapter_summary = original.load_teacher_adapter(
        transformer,
        args.adapter,
        require_reference_position=True,
    )
    output_latent = original.run_denoising(
        transformer,
        original.build_teacher_scheduler(cfg),
        initial_noise,
        prompt_embeddings,
        negative_prompt_embeddings,
        source_latent.to(torch.bfloat16),
        reference_latent.to(torch.bfloat16),
        steps=int(cfg.inference.steps),
        guidance_scale=float(cfg.inference.guidance_scale),
        label=INFERENCE_VARIANTS[0],
    )

    del transformer, text_encoder, prompt_embeddings, negative_prompt_embeddings
    del source_latent, reference_latent, initial_noise
    torch.cuda.empty_cache()
    output_video = vae.decode(output_latent, device=device, tiled=True)
    frames = original.tensor_to_uint8_frames(output_video)
    expected_shape = (65, 928, 512, 3)
    if frames.shape != expected_shape:
        raise RuntimeError(f"LoRA output shape {frames.shape}, expected {expected_shape}")
    original.save_mp4(frames, args.output_file, fps=16.0)

    result = {
        "status": "passed",
        "variant": INFERENCE_VARIANTS[0],
        "model_family": "Wan2.2-5B with Scheme4 V4 adapter",
        "base_transformer": str(TRANSFORMER_PATH),
        "sample_id": payload["sample_id"],
        "source_video": payload["source_video"],
        "target_video": payload["target_video"],
        "target_usage": payload["target_usage"],
        "reference_image": payload["reference_image"],
        "prompt": payload["prompt"],
        "generation_conditioning": payload["generation_conditioning"],
        "width": 512,
        "height": 928,
        "num_frames": 65,
        "fps": 16,
        "scheduler": "UniPCMultistepScheduler",
        "steps": 40,
        "guidance_scale": 5.0,
        "flow_shift": 3.0,
        "seed": 42,
        "physical_gpu": preflight_record["physical_gpu"],
        "adapter": str(args.adapter),
        "adapter_sha256": preflight_record["adapter_sha256"],
        "adapter_summary": adapter_summary,
        "output": str(args.output_file),
        "output_sha256": sha256_file(args.output_file),
        "preflight": preflight_record,
    }
    _atomic_json(args.run_manifest, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.preflight_only:
        initialize_cuda_for_inference()
    payload, record = preflight(args)
    if args.preflight_only:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    result = run_inference(args, payload, record)
    print(json.dumps({"status": result["status"], "output": result["output"]}))


if __name__ == "__main__":
    main()
