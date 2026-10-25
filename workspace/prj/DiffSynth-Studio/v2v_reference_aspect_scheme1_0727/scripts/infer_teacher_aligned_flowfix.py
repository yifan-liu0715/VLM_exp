#!/usr/bin/env python3
"""Generate an auditable teacher-aligned Base vs final-adapter V2V comparison."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterable

from diffusers import UniPCMultistepScheduler
import numpy as np
from omegaconf import OmegaConf
from peft import LoraConfig
from PIL import Image, ImageDraw
from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

from teacher_aligned_dataset import (
    TeacherAlignedPairedDataset,
    target_size,
)
from teacher_aligned_system import (
    ConditionAttnProcessor2_0,
    WanTransformer3DModel,
    build_timestep_sequence,
    load_text_encoder,
    load_vae,
)
from training_contracts import explicit_nonzero_visible_gpus


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "train_teacher_aligned_20000.yaml"
DEFAULT_ADAPTER = (
    EXPERIMENT_ROOT
    / "recommended/final_adapter/v2v_lora_and_source_patch.safetensors"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "validation" / "recommended"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_row(
    manifest_path: str | Path,
    sample_id: str,
) -> dict[str, str]:
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("sample_id") == sample_id
        ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one row for {sample_id}, found {len(matches)}"
        )
    row = matches[0]
    for field in ("input_video", "output_video", "reference_image"):
        if not Path(row[field]).is_file():
            raise FileNotFoundError(f"{field} does not exist: {row[field]}")
    return row


def load_aligned_sample(
    row: dict[str, str],
    *,
    num_frames: int,
    max_long_side: int,
    bucket_align: int,
) -> dict[str, object]:
    from decord import VideoReader, cpu

    source_reader = VideoReader(row["input_video"], ctx=cpu(0), num_threads=1)
    target_reader = VideoReader(row["output_video"], ctx=cpu(0), num_threads=1)
    available = min(
        len(source_reader),
        len(target_reader),
        int(row["num_frames"]),
    )
    if available < num_frames:
        raise ValueError(
            f"{row['sample_id']} has {available} paired frames, need {num_frames}"
        )
    # Start at zero for deterministic, aligned Base/adapter comparison.
    indices = list(range(num_frames))
    target_array = target_reader.get_batch(indices).asnumpy()
    height, width = target_size(
        int(target_array.shape[1]),
        int(target_array.shape[2]),
        max_long_side,
        bucket_align,
    )
    source_array = source_reader.get_batch(indices).asnumpy()
    source = TeacherAlignedPairedDataset._resize_video(
        source_array,
        height,
        width,
    )
    target = TeacherAlignedPairedDataset._resize_video(
        target_array,
        height,
        width,
    )
    reference = TeacherAlignedPairedDataset._resize_reference(
        row["reference_image"],
        height,
        width,
    )
    return {
        "source": source.unsqueeze(0),
        "target": target.unsqueeze(0),
        "reference": reference.unsqueeze(0).unsqueeze(2),
        "height": height,
        "width": width,
        "fps": float(source_reader.get_avg_fps()),
    }


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: str,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    with torch.no_grad():
        embeddings = text_encoder(ids, mask)
    return embeddings * mask.unsqueeze(-1).to(embeddings.dtype)


def token_count(transformer: torch.nn.Module, latent: torch.Tensor) -> int:
    patch_t, patch_h, patch_w = transformer.config.patch_size
    return (
        latent.shape[2] // patch_t
        * (latent.shape[3] // patch_h)
        * (latent.shape[4] // patch_w)
    )


def make_seeded_noise(
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def build_teacher_scheduler(cfg):
    """Build the official Wan2.2 UniPC flow-prediction scheduler."""
    return UniPCMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        solver_order=2,
        prediction_type="flow_prediction",
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        use_flow_sigmas=True,
        flow_shift=float(cfg.inference.flow_shift),
        timestep_spacing="linspace",
        final_sigmas_type="zero",
        use_dynamic_shifting=False,
        time_shift_type="exponential",
    )


def run_denoising(
    transformer: torch.nn.Module,
    scheduler,
    initial_noise: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    negative_prompt_embeddings: torch.Tensor,
    source_latent: torch.Tensor,
    reference_latent: torch.Tensor,
    *,
    steps: int,
    guidance_scale: float,
    label: str,
) -> torch.Tensor:
    latents = initial_noise.clone()
    scheduler.set_timesteps(steps, device=latents.device)
    main_tokens = token_count(transformer, latents)
    source_tokens = token_count(transformer, source_latent)
    reference_tokens = token_count(transformer, reference_latent)
    conditional_attention_kwargs = {
        "encoder_contion_states": source_latent,
        "encoder_first_states": reference_latent,
    }
    # CFG removes only text. Source and reference conditions remain identical.
    unconditional_attention_kwargs = conditional_attention_kwargs
    started = time.time()
    for index, scheduler_timestep in enumerate(scheduler.timesteps):
        scalar_timestep = torch.as_tensor(
            scheduler_timestep,
            device=latents.device,
            dtype=torch.float32,
        ).reshape(1)
        timestep_sequence = build_timestep_sequence(
            scalar_timestep,
            main_tokens,
            source_tokens,
            reference_tokens,
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            with transformer.cache_context("cond"):
                conditional_prediction = transformer(
                    hidden_states=latents.to(torch.bfloat16),
                    encoder_hidden_states=prompt_embeddings.to(torch.bfloat16),
                    timestep=timestep_sequence,
                    attention_kwargs=conditional_attention_kwargs,
                    return_dict=False,
                )[0]
            with transformer.cache_context("uncond"):
                unconditional_prediction = transformer(
                    hidden_states=latents.to(torch.bfloat16),
                    encoder_hidden_states=negative_prompt_embeddings.to(
                        torch.bfloat16
                    ),
                    timestep=timestep_sequence,
                    attention_kwargs=unconditional_attention_kwargs,
                    return_dict=False,
                )[0]
            prediction = unconditional_prediction + guidance_scale * (
                conditional_prediction - unconditional_prediction
            )
        latents = scheduler.step(
            prediction.float(),
            scheduler_timestep,
            latents,
            return_dict=False,
        )[0]
        print(
            f"[{label}] {index + 1:02d}/{steps} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    if not torch.isfinite(latents).all():
        raise FloatingPointError(f"{label} produced non-finite latents")
    return latents


def summarize_adapter_keys(keys: Iterable[str]) -> dict[str, int]:
    summary = {"lora": 0, "source_patch": 0, "other": 0}
    for key in keys:
        if "lora_" in key:
            summary["lora"] += 1
        elif "patch_embedding_extra" in key:
            summary["source_patch"] += 1
        else:
            summary["other"] += 1
    return summary


def load_teacher_adapter(
    transformer: torch.nn.Module,
    adapter_path: str | Path,
) -> dict[str, int]:
    raw = load_file(str(adapter_path), device="cpu")
    summary = summarize_adapter_keys(raw.keys())
    expected = {"lora": 480, "source_patch": 2, "other": 0}
    if summary != expected:
        raise ValueError(
            f"unexpected final adapter tensor contract: {summary}; "
            f"expected {expected}"
        )
    state = {
        key.removeprefix("transformer."): value
        for key, value in raw.items()
    }
    current = transformer.state_dict()
    unknown = [key for key in state if key not in current]
    mismatched = [
        key
        for key, value in state.items()
        if key in current and current[key].shape != value.shape
    ]
    if unknown or mismatched:
        raise RuntimeError(
            f"adapter mismatch: unknown={unknown[:5]}, "
            f"mismatched={mismatched[:5]}"
        )
    transformer.load_state_dict(state, strict=False)
    print(f"[adapter] loaded {summary} from {adapter_path}", flush=True)
    return summary


def tensor_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"expected 1,3,F,H,W video, got {tuple(video.shape)}")
    return (
        video.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )


def labeled(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 260), 36), fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255))
    return np.asarray(image)


def make_comparison(
    source: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    adapter: np.ndarray,
) -> np.ndarray:
    if not (source.shape == target.shape == base.shape == adapter.shape):
        raise ValueError("source, target, base, and adapter shapes must match")
    output = []
    for source_frame, target_frame, base_frame, adapter_frame in zip(
        source,
        target,
        base,
        adapter,
    ):
        top = np.concatenate(
            (
                labeled(source_frame, "SOURCE"),
                labeled(target_frame, "TARGET"),
            ),
            axis=1,
        )
        bottom = np.concatenate(
            (
                labeled(base_frame, "BASE (untrained V2V interface)"),
                labeled(adapter_frame, "TEACHER-ALIGNED ADAPTER"),
            ),
            axis=1,
        )
        output.append(np.concatenate((top, bottom), axis=0))
    return np.stack(output).astype(np.uint8, copy=False)


def save_mp4(frames: np.ndarray, path: Path, fps: float) -> None:
    from diffsynth.utils.data import save_video

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.mp4")
    save_video(
        frames,
        str(temporary),
        fps=fps,
        quality=8,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    manifest = args.manifest or Path(cfg.data.test_manifest)
    row = load_manifest_row(manifest, args.sample_id)
    if int(cfg.data.num_frames) != 97:
        raise ValueError("teacher-aligned inference requires exactly 97 frames")
    if int(cfg.inference.steps) != 50:
        raise ValueError("teacher-aligned inference requires exactly 50 steps")
    if float(cfg.inference.guidance_scale) != 5.0:
        raise ValueError("teacher-aligned inference requires CFG 5.0")
    if float(cfg.inference.flow_shift) != 5.0:
        raise ValueError("teacher-aligned inference requires flow shift 5.0")
    if int(cfg.inference.seed) != 42:
        raise ValueError("teacher-aligned inference requires seed 42")
    if not args.adapter.is_file():
        raise FileNotFoundError(f"final adapter does not exist: {args.adapter}")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "sample_id": args.sample_id,
                    "split": row["source_split"],
                    "source": row["input_video"],
                    "target": row["output_video"],
                    "reference": row["reference_image"],
                    "adapter": str(args.adapter),
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
    torch.manual_seed(int(cfg.inference.seed))
    np.random.seed(int(cfg.inference.seed))
    sample = load_aligned_sample(
        row,
        num_frames=int(cfg.data.num_frames),
        max_long_side=int(cfg.data.max_long_side),
        bucket_align=int(cfg.data.bucket_align),
    )

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
    transformer = transformer.eval().to(device)

    source_tensor = sample["source"].to(device)
    reference_tensor = sample["reference"].to(device)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        source_latent = vae.model.encode(source_tensor.float(), vae.scale)
        reference_latent = vae.model.encode(reference_tensor.float(), vae.scale)
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
        int(cfg.inference.seed),
        device,
    )

    base_latent = run_denoising(
        transformer,
        build_teacher_scheduler(cfg),
        initial_noise,
        prompt_embeddings,
        negative_prompt_embeddings,
        source_latent.to(torch.bfloat16),
        reference_latent.to(torch.bfloat16),
        steps=int(cfg.inference.steps),
        guidance_scale=float(cfg.inference.guidance_scale),
        label="base",
    )
    adapter_summary = load_teacher_adapter(transformer, args.adapter)
    adapter_latent = run_denoising(
        transformer,
        build_teacher_scheduler(cfg),
        initial_noise,
        prompt_embeddings,
        negative_prompt_embeddings,
        source_latent.to(torch.bfloat16),
        reference_latent.to(torch.bfloat16),
        steps=int(cfg.inference.steps),
        guidance_scale=float(cfg.inference.guidance_scale),
        label="adapter",
    )

    del transformer, text_encoder, prompt_embeddings
    del negative_prompt_embeddings, source_latent, reference_latent
    torch.cuda.empty_cache()
    base_video = vae.decode(base_latent, device=device, tiled=True)
    adapter_video = vae.decode(adapter_latent, device=device, tiled=True)
    videos = {
        "source": tensor_to_uint8_frames(sample["source"]),
        "target": tensor_to_uint8_frames(sample["target"]),
        "base": tensor_to_uint8_frames(base_video),
        "adapter": tensor_to_uint8_frames(adapter_video),
    }
    expected_shape = (
        int(cfg.data.num_frames),
        int(sample["height"]),
        int(sample["width"]),
        3,
    )
    for name, frames in videos.items():
        if frames.shape != expected_shape:
            raise RuntimeError(
                f"unexpected {name} shape {frames.shape}; expected {expected_shape}"
            )
    videos["comparison"] = make_comparison(
        videos["source"],
        videos["target"],
        videos["base"],
        videos["adapter"],
    )
    output_paths = {}
    for name, frames in videos.items():
        output_path = args.output_dir / "videos" / (
            f"{args.sample_id}_{name}.mp4"
        )
        save_mp4(frames, output_path, float(sample["fps"]))
        output_paths[name] = str(output_path)

    report = {
        "sample_id": args.sample_id,
        "split": row["source_split"],
        "source_video": row["input_video"],
        "target_video": row["output_video"],
        "reference_image": row["reference_image"],
        "prompt": row["caption"],
        "height": int(sample["height"]),
        "width": int(sample["width"]),
        "num_frames": int(cfg.data.num_frames),
        "fps": float(sample["fps"]),
        "scheduler": "UniPCMultistepScheduler",
        "flow_shift": float(cfg.inference.flow_shift),
        "steps": int(cfg.inference.steps),
        "guidance_scale": float(cfg.inference.guidance_scale),
        "seed": int(cfg.inference.seed),
        "physical_gpu": visible[0],
        "base_definition": "untrained V2V condition interface",
        "adapter": str(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "adapter_summary": adapter_summary,
        "identical_initial_noise": True,
        "identical_source_reference_conditions": True,
        "outputs": output_paths,
        "quality_metrics_computed": False,
    }
    report_path = (
        args.output_dir
        / "reports"
        / f"{args.sample_id}_run_manifest.json"
    )
    atomic_json(report_path, report)
    print(f"[done] comparison={output_paths['comparison']}", flush=True)
    print(f"[done] manifest={report_path}", flush=True)


if __name__ == "__main__":
    main()
