#!/usr/bin/env python3
"""Generate deterministic base-vs-LoRA Wan2.2-TI2V-5B validation videos."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
PRETRAIN_ROOT = PROJECT_ROOT / "pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B"
TEXT_ENCODER_PATH = PRETRAIN_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"
DIT_PATHS = [
    PRETRAIN_ROOT / "diffusion_pytorch_model-00001-of-00003.safetensors",
    PRETRAIN_ROOT / "diffusion_pytorch_model-00002-of-00003.safetensors",
    PRETRAIN_ROOT / "diffusion_pytorch_model-00003-of-00003.safetensors",
]
VAE_PATH = PRETRAIN_ROOT / "Wan2.2_VAE.pth"
TOKENIZER_PATH = PRETRAIN_ROOT / "google/umt5-xxl"
LORA_PATH = PROJECT_ROOT / "output_models/wan2.2_lora_quick/epoch-1.safetensors"
METADATA_PATH = PROJECT_ROOT / "data/dataset_video_train/metadata_train.csv"
DATASET_ROOT = Path("/nvmedata/workspace2/users/nina/dataset_video_train")
OUTPUT_ROOT = PROJECT_ROOT / "validation_results"

DEFAULT_SAMPLE_IDS = ["mymotion__002", "mymotion__012", "mymotion__015"]
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

HEIGHT = 320
WIDTH = 576
NUM_FRAMES = 33
SEED = 42
FPS = 15
BUCKET_RESOLUTIONS = {
    "1280x512": (256, 640),
    "1280x704": (320, 576),
    "704x1280": (576, 320),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and LoRA Wan2.2-TI2V-5B video generation."
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=DEFAULT_SAMPLE_IDS,
        help="Metadata sample IDs to validate, in generation order.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optionally limit the selected sample IDs for a quick validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already exist.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=LORA_PATH,
        help="Absolute LoRA checkpoint path.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=1.0,
        help="LoRA fusion strength.",
    )
    parser.add_argument(
        "--lora-label",
        type=str,
        default=None,
        help="Output filename label; inferred from checkpoint and alpha when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Absolute output directory for videos and the run manifest.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate sample metadata and files without loading the model or generating videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Deterministic generation seed.",
    )
    parser.add_argument(
        "--resolution-mode",
        choices=("fixed", "bucket"),
        default="fixed",
        help="Use fixed 576x320 output or aspect-preserving per-bucket resolutions.",
    )
    return parser.parse_args()


def require_paths(lora_path: Path) -> None:
    required = [
        PROJECT_ROOT,
        TEXT_ENCODER_PATH,
        *DIT_PATHS,
        VAE_PATH,
        TOKENIZER_PATH,
        lora_path,
        METADATA_PATH,
        DATASET_ROOT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def infer_lora_label(lora_path: Path, alpha: float, explicit_label: str | None) -> str:
    if explicit_label:
        label = explicit_label
    elif lora_path == LORA_PATH and alpha == 1.0:
        label = "lora"
    else:
        checkpoint = lora_path.stem.replace("-", "")
        alpha_text = f"{alpha:.2f}".replace(".", "p")
        label = f"lora_{checkpoint}_a{alpha_text}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"Invalid --lora-label: {label!r}")
    return label


def load_samples(sample_ids: list[str]) -> list[dict[str, str]]:
    wanted = set(sample_ids)
    rows_by_id: dict[str, dict[str, str]] = {}
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = Path(row["video"]).stem
            if sample_id in wanted:
                reference_entry = Path(row["reference_image"])
                reference_path = (
                    reference_entry
                    if reference_entry.is_absolute()
                    else DATASET_ROOT / reference_entry
                )
                if not reference_path.is_file():
                    raise FileNotFoundError(
                        f"Reference image for {sample_id} does not exist: {reference_path}"
                    )
                rows_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "prompt": row["prompt"],
                    "reference_image": str(reference_path),
                    "bucket": Path(row["video"]).parts[0],
                }

    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
    if missing:
        raise ValueError(f"Sample IDs not found in metadata: {missing}")
    return [rows_by_id[sample_id] for sample_id in sample_ids]


def build_pipeline() -> WanVideoPipeline:
    model_configs = [
        ModelConfig(path=str(TEXT_ENCODER_PATH)),
        ModelConfig(path=[str(path) for path in DIT_PATHS]),
        ModelConfig(path=str(VAE_PATH)),
    ]
    tokenizer_config = ModelConfig(path=str(TOKENIZER_PATH))
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )


def sample_resolution(sample: dict[str, str], mode: str) -> tuple[int, int]:
    if mode == "fixed":
        return HEIGHT, WIDTH
    bucket = sample["bucket"]
    if bucket not in BUCKET_RESOLUTIONS:
        raise ValueError(f"No aspect-preserving resolution configured for bucket {bucket!r}")
    return BUCKET_RESOLUTIONS[bucket]


def generate_video(
    pipe: WanVideoPipeline,
    sample: dict[str, str],
    variant: str,
    overwrite: bool,
    output_root: Path,
    seed: int,
    height: int,
    width: int,
) -> Path:
    output_path = output_root / f"{sample['sample_id']}_{variant}.mp4"
    if output_path.exists() and not overwrite:
        print(f"[skip] Output already exists: {output_path}", flush=True)
        return output_path

    print(
        f"[generate] {sample['sample_id']} variant={variant} seed={seed} "
        f"size={width}x{height} frames={NUM_FRAMES}",
        flush=True,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with Image.open(sample["reference_image"]) as image:
        input_image = image.convert("RGB")
        with torch.inference_mode():
            frames = pipe(
                prompt=sample["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                input_image=input_image,
                height=height,
                width=width,
                num_frames=NUM_FRAMES,
                seed=seed,
                tiled=True,
            )
    save_video(frames, str(output_path), fps=FPS, quality=5)
    print(f"[saved] {output_path}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return output_path


def main() -> None:
    args = parse_args()
    lora_path = args.lora_path.resolve()
    output_root = args.output_dir.resolve()
    require_paths(lora_path)
    lora_label = infer_lora_label(lora_path, args.lora_alpha, args.lora_label)

    selected_ids = args.sample_ids
    if args.num_samples is not None:
        if args.num_samples < 1:
            raise ValueError("--num-samples must be at least 1")
        selected_ids = selected_ids[: args.num_samples]
    samples = load_samples(selected_ids)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "samples": samples,
                    "resolution_mode": args.resolution_mode,
                    "lora": str(lora_path),
                    "output_dir": str(output_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"[device] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[samples] {', '.join(selected_ids)}")
    print("[load] Loading Wan2.2-TI2V-5B base pipeline", flush=True)
    pipe = build_pipeline()

    outputs: list[dict[str, str]] = []
    for sample in samples:
        height, width = sample_resolution(sample, args.resolution_mode)
        path = generate_video(
            pipe,
            sample,
            "base",
            args.overwrite,
            output_root,
            args.seed,
            height,
            width,
        )
        outputs.append({"sample_id": sample["sample_id"], "variant": "base", "path": str(path)})

    print(
        f"[lora] Fusing LoRA into DiT: {lora_path} alpha={args.lora_alpha} "
        f"label={lora_label}",
        flush=True,
    )
    pipe.load_lora(pipe.dit, str(lora_path), alpha=args.lora_alpha)
    for sample in samples:
        height, width = sample_resolution(sample, args.resolution_mode)
        path = generate_video(
            pipe,
            sample,
            lora_label,
            args.overwrite,
            output_root,
            args.seed,
            height,
            width,
        )
        outputs.append({"sample_id": sample["sample_id"], "variant": lora_label, "path": str(path)})

    manifest = {
        "model": "Wan2.2-TI2V-5B",
        "lora": str(lora_path),
        "lora_alpha": args.lora_alpha,
        "lora_label": lora_label,
        "parameters": {
            "resolution_mode": args.resolution_mode,
            "height": HEIGHT if args.resolution_mode == "fixed" else None,
            "width": WIDTH if args.resolution_mode == "fixed" else None,
            "sample_resolutions": {
                sample["sample_id"]: {
                    "height": sample_resolution(sample, args.resolution_mode)[0],
                    "width": sample_resolution(sample, args.resolution_mode)[1],
                }
                for sample in samples
            },
            "num_frames": NUM_FRAMES,
            "seed": args.seed,
            "tiled": True,
            "fps": FPS,
        },
        "outputs": outputs,
    }
    manifest_name = (
        "validation_manifest.json"
        if lora_label == "lora"
        else f"validation_manifest_{lora_label}.json"
    )
    manifest_path = output_root / manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
