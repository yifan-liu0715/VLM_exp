#!/usr/bin/env python3
"""Independently summarize a completed semantic clip export manifest."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    ids = [str(row.get("clip_id") or "") for row in rows]
    paths = [Path(str(row.get("output_video") or "")) for row in rows]
    hashes = [str(row.get("output_sha256") or "") for row in rows]
    hash_groups: dict[str, list[str]] = collections.defaultdict(list)
    for clip_id, digest in zip(ids, hashes):
        if digest:
            hash_groups[digest].append(clip_id)
    duplicate_hashes = {
        digest: clip_ids for digest, clip_ids in hash_groups.items() if len(clip_ids) > 1
    }
    missing = [str(path) for path in paths if not path.is_file()]
    unexpected = sorted(
        str(path)
        for path in (args.work_root / "clips").glob("*.mp4")
        if path.resolve() not in {candidate.resolve() for candidate in paths if candidate.is_file()}
    )
    keypoint_dir = args.work_root / "hand_keypoints"
    keypoint_files = list(keypoint_dir.glob("*_hands.json")) if keypoint_dir.is_dir() else []
    report: dict[str, Any] = {
        "manifest_count": len(rows),
        "unique_clip_id_count": len(set(ids)),
        "empty_clip_id_count": sum(not clip_id for clip_id in ids),
        "source_coverage_count": len({str(row.get("source_sha256") or "") for row in rows}),
        "export_ok_count": sum(row.get("export_status") == "ok" for row in rows),
        "export_error_count": sum(row.get("export_status") != "ok" for row in rows),
        "cache_status_counts": dict(
            sorted(collections.Counter(str(row.get("cache_status") or "missing") for row in rows).items())
        ),
        "unique_output_path_count": len(set(paths)),
        "missing_output_count": len(missing),
        "missing_output_examples": missing[:20],
        "unexpected_clip_file_count": len(unexpected),
        "unexpected_clip_file_examples": unexpected[:20],
        "valid_output_sha256_count": sum(SHA256_RE.fullmatch(digest) is not None for digest in hashes),
        "unique_output_sha256_count": len(set(hashes)),
        "duplicate_output_sha256_group_count": len(duplicate_hashes),
        "duplicate_output_sha256_row_count": sum(len(value) for value in duplicate_hashes.values()),
        "duplicate_output_sha256_examples": dict(list(duplicate_hashes.items())[:10]),
        "total_output_size_bytes": sum(path.stat().st_size for path in paths if path.is_file()),
        "duration_total_sec": round(sum(float(row.get("duration_sec") or 0.0) for row in rows), 6),
        "review_status_counts": dict(
            sorted(collections.Counter(str(row.get("review_status") or "missing") for row in rows).items())
        ),
        "semantic_feedback_verified_nonfalse_count": sum(
            row.get("semantic_feedback_verified") is not False for row in rows
        ),
        "hand_keypoint_file_count": len(keypoint_files),
        "training_run": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    hard_errors = (
        report["manifest_count"] != report["unique_clip_id_count"]
        or report["export_error_count"]
        or report["missing_output_count"]
        or report["unexpected_clip_file_count"]
        or report["valid_output_sha256_count"] != report["manifest_count"]
        or report["duplicate_output_sha256_group_count"]
        or report["semantic_feedback_verified_nonfalse_count"]
        or report["hand_keypoint_file_count"]
    )
    return 2 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
