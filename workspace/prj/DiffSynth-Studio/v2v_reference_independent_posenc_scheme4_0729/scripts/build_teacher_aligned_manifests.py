#!/usr/bin/env python3
"""Split the corrected paired-V2V manifest without changing its media mapping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
CONTRACT = "physical output_video + reference + caption -> physical input_video"
REQUIRED_COLUMNS = {
    "sample_id",
    "input_video",
    "output_video",
    "reference_image",
    "caption",
    "source_split",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"reference root is not a directory: {root}")
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pyc"
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_manifests(source_csv: Path, output_dir: Path) -> dict:
    source_csv = Path(source_csv).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    with source_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("source manifest contains no rows")

    split_rows: dict[str, list[dict[str, str]]] = {name: [] for name in SPLITS}
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise ValueError(f"line {line_number}: sample_id is empty")
        if sample_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate sample_id={sample_id}")
        seen_ids.add(sample_id)
        split = row["source_split"].strip()
        if split not in split_rows:
            raise ValueError(
                f"line {line_number}: unsupported source_split={split!r}"
            )
        split_rows[split].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for split in SPLITS:
        path = output_dir / f"{split}_manifest.csv"
        _write_csv(path, fieldnames, split_rows[split])
        output_hashes[split] = sha256_file(path)

    known_samples = {}
    for sample_id in ("mymotion__002",):
        row = next((item for item in rows if item["sample_id"] == sample_id), None)
        if row is not None:
            known_samples[sample_id] = {
                "source_video": row["input_video"],
                "target_video": row["output_video"],
                "reference_image": row["reference_image"],
                "caption": row["caption"],
                "source_split": row["source_split"],
            }

    report = {
        "contract": CONTRACT,
        "source_manifest": str(source_csv),
        "source_sha256": sha256_file(source_csv),
        "row_count": len(rows),
        "split_counts": {split: len(split_rows[split]) for split in SPLITS},
        "split_sha256": output_hashes,
        "known_samples": known_samples,
    }
    report_path = output_dir / "manifest_report.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_manifests(args.source, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
