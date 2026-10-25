#!/usr/bin/env python3
"""Validate the complete capture clip/keypoint manifest on the remote host."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_payload(path: Path, sample_id: str) -> tuple[bool, str, dict[str, Any] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable_json:{exc!r}", None
    if payload.get("summary", {}).get("sample_id") != sample_id:
        return False, "summary_id_mismatch", payload
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return False, "empty_frames", payload
    previous_time = -math.inf
    for frame in frames:
        if not isinstance(frame, dict):
            return False, "bad_frame", payload
        time_sec = frame.get("time_sec")
        if not isinstance(time_sec, (int, float)) or not math.isfinite(float(time_sec)):
            return False, "bad_timestamp", payload
        if float(time_sec) < previous_time:
            return False, "timestamps_not_monotonic", payload
        previous_time = float(time_sec)
        hands = frame.get("hands")
        if not isinstance(hands, list) or frame.get("hand_count") != len(hands):
            return False, "hand_count_mismatch", payload
        for hand in hands:
            points = hand.get("points") if isinstance(hand, dict) else None
            if not isinstance(points, list) or len(points) != 21:
                return False, "not_21_points", payload
            for point in points:
                if (
                    not isinstance(point, list)
                    or len(point) != 3
                    or any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in point
                    )
                ):
                    return False, "bad_xyz", payload
    return True, "valid", payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = load_jsonl(Path(args.manifest))
    ids = [str(row.get("sample_id", "")) for row in rows]
    errors: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    source_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        source_name = str(row.get("source_name", ""))
        source_to_ids[source_name].append(sample_id)
        clip = root / "clips" / str(row.get("clip_file", f"{sample_id}.mp4"))
        keypoint = root / "hand_keypoints" / f"{sample_id}_hands.json"
        item: dict[str, Any] = {
            "sample_id": sample_id,
            "source_name": source_name,
            "clip_exists": clip.is_file(),
            "json_exists": keypoint.is_file(),
        }
        if not clip.is_file():
            item["clip_hash_ok"] = False
            item["clip_reason"] = "missing_clip"
        else:
            actual_hash = digest(clip)
            item["clip_hash_ok"] = actual_hash == row.get("sha256")
            if not item["clip_hash_ok"]:
                item["clip_reason"] = "sha256_mismatch"
        if not keypoint.is_file():
            item["json_valid"] = False
            item["json_reason"] = "missing_json"
        else:
            valid, reason, payload = validate_payload(keypoint, sample_id)
            item["json_valid"] = valid
            item["json_reason"] = reason
            if payload is not None:
                summary = payload.get("summary", {})
                item.update({
                    "frames": len(payload.get("frames", [])),
                    "hand_visible_rate": summary.get("hand_visible_rate"),
                    "two_hand_rate": summary.get("two_hand_rate"),
                })
        item["valid"] = bool(item["clip_exists"] and item["clip_hash_ok"] and item["json_valid"])
        if not item["valid"]:
            errors.append(item)
        samples.append(item)

    duplicate_ids = sorted(sample_id for sample_id, count in Counter(ids).items() if count != 1)
    if duplicate_ids:
        errors.append({"reason": "duplicate_sample_ids", "sample_ids": duplicate_ids})
    report = {
        "manifest_count": len(rows),
        "unique_sample_id_count": len(set(ids)),
        "source_count": len(source_to_ids),
        "valid_count": sum(item["valid"] for item in samples),
        "clip_present_count": sum(item["clip_exists"] for item in samples),
        "clip_hash_ok_count": sum(item["clip_hash_ok"] for item in samples),
        "json_present_count": sum(item["json_exists"] for item in samples),
        "json_valid_count": sum(item["json_valid"] for item in samples),
        "coverage_kind_counts": dict(Counter(str(row.get("coverage_kind", "base_auto")) for row in rows)),
        "source_sample_count_distribution": dict(Counter(len(value) for value in source_to_ids)),
        "source_to_sample_ids": dict(sorted(source_to_ids.items())),
        "error_count": len(errors),
        "errors": errors,
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "manifest_count", "unique_sample_id_count", "source_count", "valid_count",
        "clip_present_count", "clip_hash_ok_count", "json_present_count",
        "json_valid_count", "error_count",
    )}, ensure_ascii=False))
    return 0 if not errors and len(rows) == len(set(ids)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
