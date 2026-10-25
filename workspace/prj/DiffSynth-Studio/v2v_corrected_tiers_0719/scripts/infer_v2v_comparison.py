#!/usr/bin/env python3
"""Generate a deterministic formal paired-V2V Base vs LoRA comparison."""

from __future__ import annotations

from collections.abc import Iterable
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch


BUCKET_TARGET_SIZES = {
    "1280x512": (256, 640),
    "1280x704": (320, 576),
    "704x1280": (576, 320),
}

PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
FORMAL_ROOT = PROJECT_ROOT / "v2v_formal_0715"
DEFAULT_MANIFEST = FORMAL_ROOT / "metadata/metadata_full_paired_0715.csv"
DEFAULT_CONFIG = FORMAL_ROOT / "resolved_train_config.yaml"
DEFAULT_ADAPTER = FORMAL_ROOT / "final_adapter/v2v_lora_and_source_patch.safetensors"
DEFAULT_OUTPUT = FORMAL_ROOT / "inference_comparison_0716"


def target_size_for_bucket(bucket: str) -> tuple[int, int]:
    try:
        return BUCKET_TARGET_SIZES[bucket]
    except KeyError as error:
        raise ValueError(f"unsupported bucket: {bucket}") from error


def normalize_adapter_key(key: str) -> str:
    prefix = "transformer."
    return key[len(prefix) :] if key.startswith(prefix) else key


def summarize_adapter_keys(keys: Iterable[str]) -> dict[str, int]:
    summary = {"lora": 0, "source_patch": 0, "other": 0}
    for key in keys:
        if ".lora_" in key:
            summary["lora"] += 1
        elif key.startswith("transformer.patch_embedding_extra."):
            summary["source_patch"] += 1
        else:
            summary["other"] += 1
    return summary


def validate_adapter_summary(summary: dict[str, int]) -> None:
    expected = {"lora": 480, "source_patch": 2, "other": 0}
    if summary != expected:
        raise ValueError(
            f"unexpected formal adapter tensor summary: {summary}; expected {expected}"
        )


def load_manifest_row(
    manifest_path: str | Path,
    sample_id: str,
    *,
    require_files: bool = True,
) -> dict[str, str]:
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as handle:
        matches = [row for row in csv.DictReader(handle) if row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row for {sample_id}, found {len(matches)}")
    row = matches[0]
    if require_files:
        for field in ("input_video", "output_video", "reference_image"):
            path = Path(row[field])
            if not path.is_file():
                raise FileNotFoundError(f"{field} does not exist: {path}")
    return row


def make_seeded_noise(
    shape: tuple[int, ...], seed: int, device: str | torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)


def tensor_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"expected video shape 1,3,F,H,W, got {tuple(video.shape)}")
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


def _labeled(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 220), 34), fill=(0, 0, 0))
    draw.text((10, 9), label, fill=(255, 255, 255))
    return np.asarray(image)


def make_four_panel(
    source: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    lora: np.ndarray,
) -> np.ndarray:
    arrays = [source, target, base, lora]
    if any(array.shape != source.shape for array in arrays[1:]):
        raise ValueError("all four videos must have identical shapes")
    if source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError(f"expected F,H,W,3 videos, got {source.shape}")
    frames = []
    for source_frame, target_frame, base_frame, lora_frame in zip(*arrays):
        top = np.concatenate(
            (_labeled(source_frame, "SOURCE"), _labeled(target_frame, "TARGET")), axis=1
        )
        bottom = np.concatenate(
            (_labeled(base_frame, "BASE"), _labeled(lora_frame, "LORA")), axis=1
        )
        frames.append(np.concatenate((top, bottom), axis=0))
    return np.stack(frames).astype(np.uint8, copy=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_nonzero_visible_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if len(entries) != 1 or not entries[0].isdigit() or entries[0] == "0":
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly one physical nonzero GPU, for example 2"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return entries[0]


def load_aligned_sample(row: dict[str, str], sample_frames: int) -> dict[str, object]:
    from decord import VideoReader, cpu

    source_reader = VideoReader(row["input_video"], ctx=cpu(0), num_threads=1)
    target_reader = VideoReader(row["output_video"], ctx=cpu(0), num_threads=1)
    available = min(len(source_reader), len(target_reader), int(row["num_frames"]))
    if available < sample_frames:
        raise ValueError(f"only {available} aligned frames, need {sample_frames}")
    indices = list(range(sample_frames))
    height, width = target_size_for_bucket(row["bucket"])

    formal_scripts = FORMAL_ROOT / "scripts"
    if str(formal_scripts) not in sys.path:
        sys.path.insert(0, str(formal_scripts))
    from paired_v2v_dataset import _load_reference, _resize_video

    source = _resize_video(source_reader.get_batch(indices).asnumpy(), height, width)
    target = _resize_video(target_reader.get_batch(indices).asnumpy(), height, width)
    reference = _load_reference(row["reference_image"], height, width)
    return {
        "source": source.unsqueeze(0),
        "target": target.unsqueeze(0),
        "reference": reference.unsqueeze(0).unsqueeze(2),
        "height": height,
        "width": width,
        "fps": float(source_reader.get_avg_fps()),
    }


def token_count(transformer: torch.nn.Module, latent: torch.Tensor) -> int:
    patch_t, patch_h, patch_w = transformer.config.patch_size
    return (
        latent.shape[2] // patch_t
        * (latent.shape[3] // patch_h)
        * (latent.shape[4] // patch_w)
    )


def encode_prompt(tokenizer, text_encoder, prompt: str, device: torch.device) -> torch.Tensor:
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
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = text_encoder(ids, mask)
    return embeddings * mask.unsqueeze(-1).to(embeddings.dtype)


def run_denoising(
    transformer: torch.nn.Module,
    scheduler,
    initial_noise: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    source_latent: torch.Tensor,
    reference_latent: torch.Tensor,
    *,
    steps: int,
    label: str,
) -> torch.Tensor:
    formal_scripts = FORMAL_ROOT / "scripts"
    if str(formal_scripts) not in sys.path:
        sys.path.insert(0, str(formal_scripts))
    from v2v_components import build_timestep_sequence

    scheduler.set_timesteps(steps)
    latents = initial_noise.clone()
    main_tokens = token_count(transformer, latents)
    source_tokens = token_count(transformer, source_latent)
    reference_tokens = token_count(transformer, reference_latent)
    started = time.time()
    for index, scheduler_timestep in enumerate(scheduler.timesteps):
        timestep = torch.tensor(
            [float(scheduler_timestep)], device=latents.device, dtype=torch.float32
        )
        timestep_sequence = build_timestep_sequence(
            timestep, main_tokens, source_tokens, reference_tokens
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeddings,
                timestep=timestep_sequence,
                attention_kwargs={
                    "encoder_contion_states": source_latent,
                    "encoder_first_states": reference_latent,
                },
                return_dict=False,
            )[0]
        latents = scheduler.step(
            prediction.float(), scheduler_timestep, latents, to_final=index == steps - 1
        )
        print(
            f"[{label}] step {index + 1:02d}/{steps} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    if not torch.isfinite(latents).all():
        raise FloatingPointError(f"{label} produced non-finite latents")
    return latents


def load_formal_adapter(transformer: torch.nn.Module, adapter_path: str | Path) -> dict[str, int]:
    from safetensors.torch import load_file

    raw_state = load_file(str(adapter_path), device="cpu")
    summary = summarize_adapter_keys(raw_state.keys())
    validate_adapter_summary(summary)
    state = {normalize_adapter_key(key): value for key, value in raw_state.items()}
    current = transformer.state_dict()
    unknown = [key for key in state if key not in current]
    mismatched = [
        key for key in state if key in current and current[key].shape != state[key].shape
    ]
    if unknown or mismatched:
        raise RuntimeError(
            f"adapter mismatch: unknown={unknown[:5]} mismatched={mismatched[:5]}"
        )
    transformer.load_state_dict(state, strict=False)
    print(f"[adapter] loaded {summary}", flush=True)
    return summary


def save_mp4(frames: np.ndarray, path: Path, fps: float) -> None:
    from diffsynth.utils.data import save_video

    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(path), fps=fps, quality=8, ffmpeg_params=["-pix_fmt", "yuv420p"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="mymotion__002")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = load_manifest_row(args.manifest, args.sample_id)
    if args.num_frames != 33:
        raise ValueError("formal comparison requires exactly 33 frames")
    from safetensors import safe_open

    with safe_open(str(args.adapter), framework="pt", device="cpu") as handle:
        adapter_summary = summarize_adapter_keys(handle.keys())
    validate_adapter_summary(adapter_summary)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "sample_id": args.sample_id,
                    "bucket": row["bucket"],
                    "size_hw": target_size_for_bucket(row["bucket"]),
                    "adapter_summary": adapter_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    physical_gpu = require_nonzero_visible_gpu()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sample = load_aligned_sample(row, args.num_frames)

    formal_scripts = FORMAL_ROOT / "scripts"
    if str(formal_scripts) not in sys.path:
        sys.path.insert(0, str(formal_scripts))
    from omegaconf import OmegaConf
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from models.wan2.attn_process import ConditionAttnProcessor2_0
    from models.wan2.transformer_wan import WanTransformer3DModel
    from train_v2v_lora import load_text_encoder, load_vae
    from wan_flow_scheduler import FlowMatchScheduler

    config = OmegaConf.load(args.config)
    print(f"[run] physical_gpu={physical_gpu} sample={args.sample_id}", flush=True)
    print(
        f"[run] source={row['input_video']} target={row['output_video']} "
        f"reference={row['reference_image']}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer, local_files_only=True)
    text_encoder = load_text_encoder(config.model.text_encoder).eval().requires_grad_(False).to(device)
    vae = load_vae(config.model.vae).eval().requires_grad_(False).to(device)
    transformer = WanTransformer3DModel.from_pretrained(
        config.model.transformer,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=int(config.lora.rank),
            lora_alpha=int(config.lora.alpha),
            lora_dropout=float(config.lora.dropout),
            init_lora_weights=True,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    for block in transformer.blocks:
        block.attn1.set_processor(ConditionAttnProcessor2_0())
    transformer.patch_embedding_extra = copy.deepcopy(transformer.patch_embedding).requires_grad_(False)
    transformer = transformer.eval().to(device)

    source_tensor = sample["source"].to(device)
    reference_tensor = sample["reference"].to(device)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        source_latent = vae.model.encode(source_tensor.float(), vae.scale)
        reference_latent = vae.model.encode(reference_tensor.float(), vae.scale)
    prompt_embeddings = encode_prompt(tokenizer, text_encoder, row["caption"], device)
    initial_noise = make_seeded_noise(
        tuple(source_latent.shape), seed=args.seed, device=device
    )
    scheduler = FlowMatchScheduler(
        shift=float(config.training.flow_shift), sigma_min=0.0, extra_one_step=True
    )

    base_latent = run_denoising(
        transformer,
        scheduler,
        initial_noise,
        prompt_embeddings,
        source_latent.to(torch.bfloat16),
        reference_latent.to(torch.bfloat16),
        steps=args.steps,
        label="base",
    )
    load_formal_adapter(transformer, args.adapter)
    lora_latent = run_denoising(
        transformer,
        scheduler,
        initial_noise,
        prompt_embeddings,
        source_latent.to(torch.bfloat16),
        reference_latent.to(torch.bfloat16),
        steps=args.steps,
        label="lora",
    )

    del transformer, text_encoder, prompt_embeddings, source_latent, reference_latent
    torch.cuda.empty_cache()
    base_decoded = vae.decode(base_latent, device=device, tiled=True)
    lora_decoded = vae.decode(lora_latent, device=device, tiled=True)
    source_frames = tensor_to_uint8_frames(sample["source"])
    target_frames = tensor_to_uint8_frames(sample["target"])
    base_frames = tensor_to_uint8_frames(base_decoded)
    lora_frames = tensor_to_uint8_frames(lora_decoded)
    for name, frames in {
        "source": source_frames,
        "target": target_frames,
        "base": base_frames,
        "lora": lora_frames,
    }.items():
        if frames.shape != (args.num_frames, sample["height"], sample["width"], 3):
            raise RuntimeError(f"unexpected {name} frame shape: {frames.shape}")
    comparison_frames = make_four_panel(
        source_frames, target_frames, base_frames, lora_frames
    )

    video_dir = args.output_dir / "videos"
    outputs = {}
    for name, frames in {
        "source": source_frames,
        "target": target_frames,
        "base": base_frames,
        "lora": lora_frames,
        "comparison": comparison_frames,
    }.items():
        path = video_dir / f"{args.sample_id}_{name}.mp4"
        save_mp4(frames, path, sample["fps"])
        outputs[name] = str(path)

    report_dir = args.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "sample_id": args.sample_id,
        "source_video": row["input_video"],
        "target_video": row["output_video"],
        "reference_image": row["reference_image"],
        "prompt": row["caption"],
        "bucket": row["bucket"],
        "width": sample["width"],
        "height": sample["height"],
        "num_frames": args.num_frames,
        "fps": sample["fps"],
        "steps": args.steps,
        "seed": args.seed,
        "guidance_scale": 1.0,
        "physical_gpu": physical_gpu,
        "base_transformer": config.model.transformer,
        "adapter": str(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "adapter_summary": adapter_summary,
        "outputs": outputs,
        "quality_metrics_computed": False,
    }
    report_path = report_dir / f"{args.sample_id}_run_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] comparison={outputs['comparison']}", flush=True)
    print(f"[done] manifest={report_path}", flush=True)


if __name__ == "__main__":
    main()
