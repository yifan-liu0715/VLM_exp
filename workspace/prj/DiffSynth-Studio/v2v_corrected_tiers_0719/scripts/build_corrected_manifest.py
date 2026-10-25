#!/usr/bin/env python3
"""Freeze and audit the corrected source/target direction for V2V training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tiered_training_utils import swap_v2v_direction


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError("source manifest is empty")
    if not {"input_video", "output_video", "reference_image", "caption"}.issubset(fieldnames):
        raise ValueError("source manifest lacks required V2V fields")

    corrected = [swap_v2v_direction(row) for row in rows]
    seen: set[str] = set()
    bucket_counts: dict[str, int] = {}
    for index, row in enumerate(corrected, start=2):
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"line {index}: duplicate sample_id {sample_id}")
        seen.add(sample_id)
        if "/output_video/" not in row["input_video"].replace("\\", "/"):
            raise ValueError(f"line {index}: corrected source is not from output_video")
        if "/input_video/" not in row["output_video"].replace("\\", "/"):
            raise ValueError(f"line {index}: corrected target is not from input_video")
        for field in ("input_video", "output_video", "reference_image"):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(f"line {index}: missing {field}: {row[field]}")
        if int(row["num_frames"]) < 33:
            raise ValueError(f"line {index}: fewer than 33 frames")
        if not row["caption"].strip():
            raise ValueError(f"line {index}: empty caption")
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corrected)
    temporary.replace(args.output)

    report = {
        "contract": "existing output_video + reference_image + caption -> existing input_video",
        "source_manifest": str(args.source),
        "corrected_manifest": str(args.output),
        "row_count": len(corrected),
        "bucket_counts": bucket_counts,
        "manifest_sha256": sha256_file(args.output),
        "sample_002": next(row for row in corrected if row["sample_id"] == "mymotion__002"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
