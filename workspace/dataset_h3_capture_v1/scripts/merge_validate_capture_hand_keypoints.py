#!/usr/bin/env python3
"""Flatten and strictly validate sharded capture hand-landmark outputs."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def validate_payload(path: Path, expected_id: str) -> tuple[bool, str, dict[str, Any] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable_json:{exc!r}", None
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
        if not isinstance(hands, list):
            return False, "bad_hands", payload
        if frame.get("hand_count") != len(hands):
            return False, "hand_count_mismatch", payload
        for hand in hands:
            points = hand.get("points") if isinstance(hand, dict) else None
            if not isinstance(points, list) or len(points) != 21:
                return False, "not_21_points", payload
            for point in points:
                if not isinstance(point, list) or len(point) != 3:
                    return False, "not_xyz", payload
                if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in point):
                    return False, "non_finite_point", payload
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("sample_id") != expected_id:
        return False, "summary_id_mismatch", payload
    return True, "valid", payload


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        try:
            target.symlink_to(source)
            return "symlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--flat-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = load_jsonl(Path(args.manifest))
    shard_dir = Path(args.shard_dir)
    flat_dir = Path(args.flat_dir)
    flat_dir.mkdir(parents=True, exist_ok=True)
    expected = {str(row["sample_id"]) for row in manifest}
    sources: dict[str, list[Path]] = {}
    for path in shard_dir.glob("shard_*/*_hands.json"):
        sample_id = path.name.removesuffix("_hands.json")
        sources.setdefault(sample_id, []).append(path)

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    link_modes: dict[str, int] = {}
    for sample_id in sorted(expected):
        candidates = sources.get(sample_id, [])
        if len(candidates) != 1:
            errors.append({"sample_id": sample_id, "reason": "missing_or_duplicate_output", "paths": [str(p) for p in candidates]})
            continue
        source = candidates[0]
        target = flat_dir / source.name
        mode = link_or_copy(source, target)
        link_modes[mode] = link_modes.get(mode, 0) + 1
        valid, reason, payload = validate_payload(target, sample_id)
        item: dict[str, Any] = {"sample_id": sample_id, "path": str(target), "valid": valid, "reason": reason}
        if payload is not None:
            summary = payload.get("summary", {})
            item.update({
                "frames": len(payload.get("frames", [])),
                "hand_visible_rate": summary.get("hand_visible_rate"),
                "two_hand_rate": summary.get("two_hand_rate"),
            })
        samples.append(item)
        if not valid:
            errors.append({"sample_id": sample_id, "reason": reason, "path": str(target)})

    unexpected = sorted(set(sources) - expected)
    for sample_id in unexpected:
        errors.append({"sample_id": sample_id, "reason": "unexpected_output", "paths": [str(p) for p in sources[sample_id]]})
    report = {
        "manifest_count": len(manifest),
        "expected_unique_count": len(expected),
        "flattened_count": len(samples),
        "valid_count": sum(item["valid"] for item in samples),
        "error_count": len(errors),
        "link_modes": link_modes,
        "errors": errors,
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("manifest_count", "flattened_count", "valid_count", "error_count")}))
    return 0 if not errors and len(samples) == len(expected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
