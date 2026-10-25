#!/usr/bin/env python3
"""Load one Scheme 3 sample per native bucket and record tensor contracts."""

from __future__ import annotations

import json
from pathlib import Path

from teacher_aligned_dataset import TeacherAlignedPairedDataset


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "validation" / "cpu_dataset_report.json"
EXPECTED = {
    "1280x512": ([3, 97, 512, 1280], [3, 512, 1280]),
    "1280x704": ([3, 97, 704, 1280], [3, 704, 1280]),
    "704x1280": ([3, 97, 1280, 704], [3, 1280, 704]),
}


def main() -> None:
    dataset = TeacherAlignedPairedDataset(
        ROOT / "metadata" / "train_manifest.csv",
        sample_frames=97,
        bucket_align=32,
        required_split="train",
    )
    first_by_bucket: dict[str, int] = {}
    for index, row in enumerate(dataset.rows):
        first_by_bucket.setdefault(row["bucket"], index)

    samples = []
    for bucket, (video_shape, reference_shape) in EXPECTED.items():
        sample = dataset[first_by_bucket[bucket]]
        source = sample["source_video"]
        target = sample["target_video"]
        reference = sample["reference_image"]
        if list(source.shape) != video_shape:
            raise RuntimeError(
                f"{bucket}: Source shape {list(source.shape)} != {video_shape}"
            )
        if list(target.shape) != video_shape:
            raise RuntimeError(
                f"{bucket}: Target shape {list(target.shape)} != {video_shape}"
            )
        if list(reference.shape) != reference_shape:
            raise RuntimeError(
                f"{bucket}: Reference shape {list(reference.shape)} "
                f"!= {reference_shape}"
            )
        ranges = {
            "source": [float(source.min()), float(source.max())],
            "target": [float(target.min()), float(target.max())],
            "reference": [float(reference.min()), float(reference.max())],
        }
        if any(
            low < -1.0 or high > 1.0
            for low, high in ranges.values()
        ):
            raise RuntimeError(f"{bucket}: tensor range outside [-1, 1]")
        samples.append(
            {
                "bucket": bucket,
                "sample_id": sample["sample_id"],
                "source_shape": list(source.shape),
                "target_shape": list(target.shape),
                "reference_shape": list(reference.shape),
                "ranges": ranges,
                "source_path": sample["source_path"],
                "target_path": sample["target_path"],
                "reference_path": sample["reference_path"],
            }
        )

    payload = {
        "status": "passed",
        "scheme": "native_resolution_scheme3",
        "source_target_preprocessing": "native pixels, normalization only",
        "reference_preprocessing": (
            "aspect fit to native video canvas, replicate minimum padding"
        ),
        "train_rows": len(dataset),
        "samples": samples,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(REPORT)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
