#!/usr/bin/env python3
"""Select a deterministic, bucket-balanced, source-diverse validation subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DATASET_ROOT = Path("/nvmedata/workspace2/users/nina/dataset_video_train")
DEFAULT_INPUT = PROJECT_ROOT / "data_splits/source_disjoint_v2/val_rich.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_splits/source_disjoint_v2/validation_subset_15.json"
BUCKETS = ("1280x512", "1280x704", "704x1280")


def source_group(sample_id: str) -> str:
    parts = sample_id.split("__")
    if len(parts) >= 3 and re.fullmatch(r"\d+", parts[-1]):
        return "__".join(parts[:-1])
    return sample_id


def stable_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open("r", encoding="utf-8", newline="")))
    selected: list[dict] = []
    used_groups: set[str] = set()
    for bucket in BUCKETS:
        candidates = [
            row
            for row in rows
            if row["bucket"] == bucket
            and row["reference_is_fallback"] == "False"
            and (DATASET_ROOT / row["reference_image"]).is_file()
            and (DATASET_ROOT / row["output_video"]).is_file()
        ]
        candidates.sort(key=lambda row: stable_key(args.seed, row["sample_id"]))
        bucket_selected = []
        for row in candidates:
            group = source_group(row["sample_id"])
            if group in used_groups:
                continue
            bucket_selected.append(row)
            used_groups.add(group)
            if len(bucket_selected) == args.per_bucket:
                break
        if len(bucket_selected) < args.per_bucket:
            selected_ids = {row["sample_id"] for row in bucket_selected}
            for row in candidates:
                if row["sample_id"] in selected_ids:
                    continue
                bucket_selected.append(row)
                selected_ids.add(row["sample_id"])
                used_groups.add(source_group(row["sample_id"]))
                if len(bucket_selected) == args.per_bucket:
                    break
        if len(bucket_selected) != args.per_bucket:
            raise RuntimeError(f"Could not select {args.per_bucket} unique groups for {bucket}")
        selected.extend(bucket_selected)

    payload = {
        "source": str(args.input),
        "seed": args.seed,
        "per_bucket": args.per_bucket,
        "criteria": "reference_is_fallback=False, files exist; maximize source-group diversity before filling each bucket",
        "samples": [
            {
                "sample_id": row["sample_id"],
                "bucket": row["bucket"],
                "source_group": source_group(row["sample_id"]),
                "caption": row["caption"],
                "reference_image": row["reference_image"],
                "output_video": row["output_video"],
            }
            for row in selected
        ],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".txt").write_text(
        "\n".join(row["sample_id"] for row in selected) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
