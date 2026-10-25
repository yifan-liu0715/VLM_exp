#!/usr/bin/env python3
"""Build a static, GPU-free SCAIL-2 Base/DPO SAM2-fallback plan."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


WAN_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
BASE_SHA256 = "8b844f026510f0b94d3209d66444b403a089ec179dc149d85513c13b2fa37118"
DPO_SHA256 = "9c317e5c1f8174ea2799b7ca1a81a36fbd80c5f0787379d9103ced3fe3b94785"
MASK_PREPROCESSOR = "official SAM2 fallback; not official SAM3"
PYTHON = Path("/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python")


def approved_mask_bundle(root: Path, sample_id: str) -> bool:
    processed = root / "preprocessed_sam2" / sample_id
    manifest_path = processed / "mask_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    names = ("ref_mask.jpg", "rendered_v2.mp4", "rendered_mask_v2.mp4")
    return (
        manifest.get("sample_id") == sample_id
        and manifest.get("status") == "passed"
        and manifest.get("mask_preprocessor") == MASK_PREPROCESSOR
        and all((processed / name).is_file() for name in names)
    )


def build_common_argv(
    root: Path, sample: dict, output_dir: Path, segment_len: int = 81
) -> list[str]:
    sample_id = sample["sample_id"]
    processed = root / "preprocessed_sam2" / sample_id
    wan = root / "code" / "SCAIL-2-wan" / f"SCAIL-2-{WAN_COMMIT}"
    return [
        str(wan / "generate.py"),
        "--model", "SCAIL-14B",
        "--ckpt_dir", str(root / "weights" / "SCAIL-2"),
        "--scail_path", str(root / "weights" / "converted" / "SCAIL-2.safetensors"),
        "--scail_config_path", str(wan / "configs" / "config-14b.json"),
        "--prompt", sample["prompt"],
        "--image", str(root / "inputs" / sample_id / "ref.jpg"),
        "--mask_image", str(processed / "ref_mask.jpg"),
        "--pose", str(processed / "rendered_v2.mp4"),
        "--mask_video", str(processed / "rendered_mask_v2.mp4"),
        "--replace_flag",
        "--target_h", str(sample["height"]),
        "--target_w", str(sample["width"]),
        "--sample_solver", "unipc",
        "--sample_steps", "40",
        "--sample_shift", "3.0",
        "--sample_guide_scale", "5.0",
        "--segment_len", str(segment_len),
        "--segment_overlap", "5",
        "--base_seed", "42",
        "--offload_model", "true",
        "--save_file", str(output_dir / sample_id / "PLACEHOLDER.mp4"),
    ]


def build_plan(root: Path, segment_len: int = 81) -> dict:
    input_manifest = json.loads(
        (root / "manifests" / "input_manifest.json").read_text(encoding="utf-8")
    )
    output_dir = root / "outputs" / "sam2_fallback_base_vs_dpo"
    runs = []
    for sample in input_manifest["samples"]:
        sample_id = sample["sample_id"]
        ready = approved_mask_bundle(root, sample_id)
        common = build_common_argv(root, sample, output_dir, segment_len=segment_len)
        for variant in ("base", "dpo"):
            argv = list(common)
            argv[-1] = str(output_dir / sample_id / f"{variant}.mp4")
            if variant == "dpo":
                argv += [
                    "--lora_path",
                    str(root / "weights" / "converted" / "bias-aware-dpo-lora.safetensors"),
                    "--lora_alpha",
                    "1.0",
                ]
            runs.append(
                {
                    "sample_id": sample_id,
                    "variant": variant,
                    "python": str(PYTHON),
                    "argv": argv,
                    "command": shlex.join([str(PYTHON), *argv]),
                    "ready": ready,
                    "mask_manifest": str(
                        root / "preprocessed_sam2" / sample_id / "mask_manifest.json"
                    ),
                }
            )
    all_ready = bool(runs) and all(run["ready"] for run in runs)
    return {
        "status": "ready" if all_ready else "blocked_on_approved_sam2_masks",
        "mask_preprocessor": MASK_PREPROCESSOR,
        "target_usage": "evaluation-only; excluded from generation conditioning",
        "wan_source_commit": WAN_COMMIT,
        "segment_len": segment_len,
        "comparison_rule": (
            "DPO differs from Base only by --lora_path and --lora_alpha; "
            "all conditioning and sampling parameters are identical."
        ),
        "base": {
            "path": str(root / "weights" / "converted" / "SCAIL-2.safetensors"),
            "sha256": BASE_SHA256,
        },
        "dpo": {
            "path": str(root / "weights" / "converted" / "bias-aware-dpo-lora.safetensors"),
            "sha256": DPO_SHA256,
            "alpha": 1.0,
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-len", type=int, default=81)
    args = parser.parse_args()
    plan = build_plan(args.root.resolve(), segment_len=args.segment_len)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": plan["status"], "runs": len(plan["runs"])}))


if __name__ == "__main__":
    main()
