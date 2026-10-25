#!/usr/bin/env python3
"""Merge isolated hand-detection shards into one canonical JSON directory."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

if __package__:
    from .causal_clip_pipeline import load_jsonl
else:
    from causal_clip_pipeline import load_jsonl


def merge_shards(rows: list[dict], shard_root: Path, output: Path) -> dict:
    expected = {str(row["clip_id"]) for row in rows}
    if len(expected) != len(rows):
        raise ValueError("manifest has duplicate clip IDs")
    found: dict[str, Path] = {}
    duplicates: list[str] = []
    unexpected: list[str] = []
    for path in shard_root.glob("shard_*/*_hands.json"):
        clip_id = path.name.removesuffix("_hands.json")
        if clip_id not in expected:
            unexpected.append(clip_id)
        elif clip_id in found:
            duplicates.append(clip_id)
        else:
            found[clip_id] = path
    missing = sorted(expected - set(found))
    report = {
        "manifest_count": len(rows), "found_count": len(found),
        "missing_count": len(missing), "duplicate_count": len(set(duplicates)),
        "unexpected_count": len(set(unexpected)), "missing_clip_ids": missing,
        "duplicate_clip_ids": sorted(set(duplicates)), "unexpected_clip_ids": sorted(set(unexpected)),
    }
    if missing or duplicates or unexpected:
        return report
    output.mkdir(parents=True, exist_ok=True)
    for clip_id, source in found.items():
        destination = output / f"{clip_id}_hands.json"
        temporary = output / f".{clip_id}.{os.getpid()}.tmp"
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = merge_shards(load_jsonl(Path(args.manifest)), Path(args.shard_root), Path(args.output))
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "manifest_count", "found_count", "missing_count", "duplicate_count", "unexpected_count"
    )}))
    return 0 if not any(report[key] for key in ("missing_count", "duplicate_count", "unexpected_count")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
