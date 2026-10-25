#!/usr/bin/env python3
import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video

PROJECT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
WEIGHTS = PROJECT / "pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B"
PAIRS = PROJECT / "data_0713/v2v_quick_0713/v2v_pairs_0713.csv"
LORA_DIR = PROJECT / "output_models/wan2.2_lora_quick_v2v_0713"
OUTPUT = PROJECT / "validation_results_0713"
HEIGHT, WIDTH, NUM_FRAMES, FPS = 320, 576, 33, 15
NEGATIVE = "low quality, blurry, distorted faces, artifacts, flicker, subtitles, watermark"


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic native input_video V2V base-vs-LoRA comparison.")
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoising-strength", type=float, default=0.45)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def latest_lora():
    checkpoints = list(LORA_DIR.glob("step-*.safetensors"))
    if not checkpoints:
        checkpoints = list(LORA_DIR.glob("epoch-*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"No LoRA checkpoint in {LORA_DIR}")
    return max(checkpoints, key=lambda path: int(path.stem.split("-")[-1]))


def read_pairs(sample_ids, num_samples):
    with PAIRS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if sample_ids:
        selected = [row for sid in sample_ids for row in rows if row["sample_id"] == sid]
        if len(selected) != len(sample_ids):
            found = {row["sample_id"] for row in selected}
            raise ValueError(f"Missing pairs: {sorted(set(sample_ids) - found)}")
        return selected
    return rows[:num_samples]


def build_pipeline():
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(WEIGHTS / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=[str(WEIGHTS / f"diffusion_pytorch_model-0000{i}-of-00003.safetensors") for i in (1, 2, 3)]),
            ModelConfig(path=str(WEIGHTS / "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(WEIGHTS / "google/umt5-xxl")),
    )


def generate(pipe, row, variant, args, output_dir):
    destination = output_dir / f"{row['sample_id']}_{variant}.mp4"
    if destination.exists() and not args.overwrite:
        print(f"[skip] {destination}", flush=True)
        return destination
    source = VideoData(row["source_video"], height=HEIGHT, width=WIDTH)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[generate] {row['sample_id']} {variant} source={row['source_video']}", flush=True)
    with torch.inference_mode():
        frames = pipe(
            prompt=row["prompt"], negative_prompt=NEGATIVE,
            input_video=source, denoising_strength=args.denoising_strength,
            height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
            seed=args.seed, tiled=True,
            num_inference_steps=args.num_inference_steps,
        )
    save_video(frames, str(destination), fps=FPS, quality=5)
    gc.collect()
    torch.cuda.empty_cache()
    return destination


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    lora_path = args.lora_path.resolve() if args.lora_path else latest_lora().resolve()
    rows = read_pairs(args.sample_ids, args.num_samples)
    required = [PAIRS, lora_path, WEIGHTS / "Wan2.2_VAE.pth"]
    required += [Path(row[key]) for row in rows for key in ("source_video", "target_video")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    preflight = {
        "lora": str(lora_path), "pairs": rows,
        "output_dir": str(output_dir),
        "parameters": {"height": HEIGHT, "width": WIDTH, "num_frames": NUM_FRAMES, "seed": args.seed, "tiled": True, "denoising_strength": args.denoising_strength, "num_inference_steps": args.num_inference_steps},
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline()
    outputs = []
    for row in rows:
        outputs.append({"sample_id": row["sample_id"], "variant": "base", "path": str(generate(pipe, row, "base", args, output_dir))})
    print(f"[lora] loading {lora_path} alpha={args.lora_alpha}", flush=True)
    pipe.load_lora(pipe.dit, str(lora_path), alpha=args.lora_alpha)
    for row in rows:
        outputs.append({"sample_id": row["sample_id"], "variant": "lora", "path": str(generate(pipe, row, "lora", args, output_dir))})
    manifest = {**preflight, "outputs": outputs, "method_note": "Native input_video V2V. The source video initializes the noisy latent; LoRA was trained on held-out-disjoint target-domain videos."}
    (output_dir / "validation_manifest_0713.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
