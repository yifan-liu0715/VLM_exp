#!/usr/bin/env python3
"""Freeze the audited 3,916-row metadata into an absolute-path formal-run manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

from v2v_components import absolutize_media_paths, validate_manifest


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DEFAULT_SOURCE = PROJECT_ROOT / "data_v2v_formal_0713/metadata_full_paired_0713.csv"
DEFAULT_DATASET_ROOT = Path("/nvmedata/workspace2/users/nina/dataset_video_train")
DEFAULT_OUTPUT = PROJECT_ROOT / "v2v_formal_0715/metadata/metadata_full_paired_0715.csv"
DEFAULT_REPORT = PROJECT_ROOT / "v2v_formal_0715/metadata/data_manifest_full_0715.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(source: Path, dataset_root: Path, output: Path, report_path: Path) -> dict:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("source metadata has no header")
        rows = [absolutize_media_paths(row, dataset_root) for row in reader]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)

    validated = validate_manifest(output, minimum_frames=33)
    if len(validated) != 3916:
        raise ValueError(f"formal manifest must contain 3916 rows, got {len(validated)}")

    report = {
        "manifest": str(output),
        "manifest_sha256": sha256(output),
        "source_manifest": str(source),
        "source_manifest_sha256": sha256(source),
        "dataset_root": str(dataset_root.resolve()),
        "sample_count": len(validated),
        "buckets": dict(sorted(Counter(row["bucket"] for row in validated).items())),
        "source_splits": dict(
            sorted(Counter(row.get("source_split", "unknown") for row in validated).items())
        ),
        "minimum_frames": 33,
        "path_mode": "absolute",
        "pair_contract": "input_video -> output_video, conditioned by reference_image and caption",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(freeze(args.source, args.dataset_root, args.output, args.report), indent=2))
