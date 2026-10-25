#!/usr/bin/env python3
"""Validate transferred causal clips and signed hand landmarks without raw sources."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .causal_clip_pipeline import load_jsonl, sha256_file, validate_keypoint_payload
else:
    from causal_clip_pipeline import load_jsonl, sha256_file, validate_keypoint_payload


def validate_derivatives(rows: list[dict], clip_dir: Path, keypoint_dir: Path) -> dict:
    errors: list[dict[str, str]] = []
    zero_hand: list[str] = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    for row in rows:
        clip_id = str(row.get("clip_id") or "")
        if not clip_id or clip_id in seen_ids:
            errors.append({"clip_id": clip_id, "reason": "missing_or_duplicate_clip_id"})
            continue
        seen_ids.add(clip_id)
        seen_sources.add(str(row.get("source_rel") or ""))
        clip = clip_dir / f"{clip_id}.mp4"
        if not clip.is_file():
            errors.append({"clip_id": clip_id, "reason": "missing_clip"})
        elif sha256_file(clip) != row.get("output_sha256"):
            errors.append({"clip_id": clip_id, "reason": "clip_sha256_mismatch"})
        json_path = keypoint_dir / f"{clip_id}_hands.json"
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append({"clip_id": clip_id, "reason": "missing_or_invalid_keypoint_json"})
            continue
        for reason in validate_keypoint_payload(payload, row, require_provenance=True):
            errors.append({"clip_id": clip_id, "reason": reason})
        frames = payload.get("frames") or []
        if frames and all(not frame.get("hands") for frame in frames):
            zero_hand.append(clip_id)
    return {
        "manifest_count": len(rows),
        "unique_clip_count": len(seen_ids),
        "source_coverage_count": len(seen_sources - {""}),
        "valid_count": len(seen_ids) - len({err["clip_id"] for err in errors}),
        "error_count": len(errors),
        "zero_hand_count": len(zero_hand),
        "zero_hand_clip_ids": zero_hand,
        "hand_keypoint_usable_count": len(seen_ids) - len({err["clip_id"] for err in errors}) - len(zero_hand),
        "zero_hand_training_status": "exclude_pending_manual_review" if zero_hand else "none",
        "review_status_counts": dict(Counter(str(row.get("review_status", "")) for row in rows)),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--keypoint-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = validate_derivatives(load_jsonl(Path(args.manifest)), Path(args.clip_dir), Path(args.keypoint_dir))
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "manifest_count", "valid_count", "error_count", "zero_hand_count", "source_coverage_count"
    )}))
    return 0 if not report["error_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
