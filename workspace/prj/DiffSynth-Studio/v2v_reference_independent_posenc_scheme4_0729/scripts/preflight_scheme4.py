#!/usr/bin/env python3
"""Validate Scheme 4 media geometry before loading the 5B model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from scheme4_geometry import scheme4_canvas
from teacher_aligned_dataset import TeacherAlignedPairedDataset


def latent_and_patch_shapes(
    *,
    height: int,
    width: int,
    num_frames: int,
) -> dict[str, list[int]]:
    if height % 32 or width % 32:
        raise ValueError("Scheme 4 latent canvas must be 32-aligned")
    latent_frames = (int(num_frames) - 1) // 4 + 1
    latent_height = int(height) // 16
    latent_width = int(width) // 16
    return {
        "video_latent": [48, latent_frames, latent_height, latent_width],
        "video_patch_grid": [
            latent_frames,
            latent_height // 2,
            latent_width // 2,
        ],
        "reference_latent": [48, 1, latent_height, latent_width],
        "reference_grid": [1, latent_height // 2, latent_width // 2],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    canvas_by_bucket = OmegaConf.to_container(
        cfg.data.canvas_by_bucket,
        resolve=True,
    )
    dataset = TeacherAlignedPairedDataset(
        cfg.data.train_manifest,
        sample_frames=int(cfg.data.num_frames),
        bucket_align=int(cfg.data.bucket_align),
        canvas_by_bucket=canvas_by_bucket,
        required_split="train",
    )
    first_by_bucket: dict[str, int] = {}
    for index, row in enumerate(dataset.rows):
        bucket = f"{int(row['width'])}x{int(row['height'])}"
        first_by_bucket.setdefault(bucket, index)
    expected_buckets = set(canvas_by_bucket)
    if set(first_by_bucket) != expected_buckets:
        raise RuntimeError(
            f"dataset buckets differ: {set(first_by_bucket)} "
            f"!= {expected_buckets}"
        )

    samples = []
    for bucket in sorted(first_by_bucket):
        sample = dataset[first_by_bucket[bucket]]
        height, width = scheme4_canvas(bucket)
        source = sample["source_video"]
        target = sample["target_video"]
        reference = sample["reference_image"]
        expected_video_shape = (3, int(cfg.data.num_frames), height, width)
        if tuple(source.shape) != expected_video_shape:
            raise RuntimeError(
                f"{bucket} Source shape {tuple(source.shape)} "
                f"!= {expected_video_shape}"
            )
        if tuple(target.shape) != expected_video_shape:
            raise RuntimeError(
                f"{bucket} Target shape {tuple(target.shape)} "
                f"!= {expected_video_shape}"
            )
        if tuple(reference.shape) != (3, height, width):
            raise RuntimeError(
                f"{bucket} Reference shape {tuple(reference.shape)} "
                f"!= {(3, height, width)}"
            )
        for name, tensor in (
            ("Source", source),
            ("Target", target),
            ("Reference", reference),
        ):
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{bucket} {name} contains nonfinite values")
            if float(tensor.min()) < -1.0001 or float(tensor.max()) > 1.0001:
                raise RuntimeError(f"{bucket} {name} is outside [-1,1]")
        shapes = latent_and_patch_shapes(
            height=height,
            width=width,
            num_frames=int(cfg.data.num_frames),
        )
        if (
            shapes["reference_grid"][0] > int(cfg.reference_position.max_frames)
            or shapes["reference_grid"][1]
            > int(cfg.reference_position.max_height)
            or shapes["reference_grid"][2]
            > int(cfg.reference_position.max_width)
        ):
            raise RuntimeError(
                f"{bucket} Reference grid exceeds position-table bounds"
            )
        samples.append(
            {
                "bucket": bucket,
                "sample_id": sample["sample_id"],
                "source_shape": list(source.shape),
                "target_shape": list(target.shape),
                "reference_shape": list(reference.shape),
                **shapes,
            }
        )

    payload = {
        "status": "passed",
        "scheme": "reference_independent_posenc_scheme4",
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
