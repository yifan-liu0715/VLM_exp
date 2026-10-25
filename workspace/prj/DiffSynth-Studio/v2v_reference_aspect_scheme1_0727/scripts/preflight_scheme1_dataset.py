#!/usr/bin/env python3
"""Load one Scheme-1 sample per native resolution and record tensor contracts."""

from __future__ import annotations

import json
from pathlib import Path

from teacher_aligned_dataset import TeacherAlignedPairedDataset


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "preflight" / "cpu_dataset_report.json"


def main() -> None:
    dataset = TeacherAlignedPairedDataset(
        ROOT / "metadata" / "train_manifest.csv",
        sample_frames=97,
        max_long_side=704,
        bucket_align=32,
        required_split="train",
    )
    first_by_bucket: dict[str, int] = {}
    for index, row in enumerate(dataset.rows):
        first_by_bucket.setdefault(row["bucket"], index)

    samples = []
    for bucket in ("1280x512", "1280x704", "704x1280"):
        sample = dataset[first_by_bucket[bucket]]
        source = sample["source_video"]
        target = sample["target_video"]
        reference = sample["reference_image"]
        if source.shape != target.shape:
            raise RuntimeError(f"{bucket}: Source and Target shapes differ")
        if reference.shape != source[:, 0].shape:
            raise RuntimeError(f"{bucket}: Reference and video frame shapes differ")
        samples.append(
            {
                "bucket": bucket,
                "sample_id": sample["sample_id"],
                "source_shape": list(source.shape),
                "target_shape": list(target.shape),
                "reference_shape": list(reference.shape),
                "source_range": [float(source.min()), float(source.max())],
                "target_range": [float(target.min()), float(target.max())],
                "reference_range": [
                    float(reference.min()),
                    float(reference.max()),
                ],
                "source_path": sample["source_path"],
                "target_path": sample["target_path"],
                "reference_path": sample["reference_path"],
            }
        )

    payload = {
        "status": "passed",
        "scheme": "reference_aspect_scheme1",
        "source_target_preprocessing": "unchanged",
        "reference_preprocessing": (
            "foreground-aware aspect crop with white-padding fallback"
        ),
        "samples": samples,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(REPORT)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
