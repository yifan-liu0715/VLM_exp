from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SampleInput:
    sample_id: str
    source_video: Path
    reference_image: Path
    prompt: str
    width: int
    height: int
    num_frames: int
    fps: float
    target_video: Path


def load_sample(root: Path, sample_id: str) -> SampleInput:
    manifest_path = root / "manifests" / "input_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        row = next(item for item in data["samples"] if item["sample_id"] == sample_id)
    except StopIteration as error:
        raise ValueError(f"sample not found: {sample_id}") from error

    sample = SampleInput(
        sample_id=str(row["sample_id"]),
        source_video=Path(row["source_video"]),
        reference_image=Path(row["reference_image"]),
        prompt=str(row["prompt"]),
        width=int(row["width"]),
        height=int(row["height"]),
        num_frames=int(row["num_frames"]),
        fps=float(row["fps"]),
        target_video=Path(row["target_video"]),
    )
    if "output_video" not in sample.source_video.parts:
        raise ValueError("approved dataset direction violated: source is not output_video")
    if "input_video" not in sample.target_video.parts:
        raise ValueError("approved dataset direction violated: target is not input_video")
    if sample.width <= 0 or sample.height <= 0 or sample.num_frames <= 0 or sample.fps <= 0:
        raise ValueError(f"invalid media metadata for {sample.sample_id}")
    return sample


def assert_conditioning_paths(sample: SampleInput, paths: Iterable[Path]) -> None:
    target = sample.target_video.resolve(strict=False)
    for candidate in paths:
        path = Path(candidate).resolve(strict=False)
        if path == target:
            raise ValueError(f"target leakage: {candidate}")
