#!/usr/bin/env python3
"""Independently audit a filtered hand-keypoint dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--excluded-id", action="append", default=[])
    parser.add_argument("--old-root-text", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    rows = load_jsonl(args.manifest)
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    by_id = {str(row.get("clip_id") or ""): row for row in rows}
    clips = {path.stem: path for path in (root / "clips").glob("*.mp4") if path.is_file()}
    keypoints = {
        path.name.removesuffix("_hands.json"): path
        for path in (root / "hand_keypoints").glob("*_hands.json")
        if path.is_file()
    }
    errors: list[dict[str, str]] = []
    expected_ids = set(by_id)
    if len(rows) != args.expected or len(by_id) != args.expected or "" in by_id:
        errors.append({"reason": "manifest_count_or_id_mismatch"})
    if set(clips) != expected_ids:
        errors.append({"reason": "clip_set_mismatch"})
    if set(keypoints) != expected_ids:
        errors.append({"reason": "keypoint_set_mismatch"})
    if validation.get("valid_count") != args.expected or validation.get("error_count") != 0:
        errors.append({"reason": "validation_report_failed"})
    if validation.get("zero_hand_count") != 0:
        errors.append({"reason": "validation_report_has_zero_hand"})
    for clip_id in args.excluded_id:
        if clip_id in expected_ids or clip_id in clips or clip_id in keypoints:
            errors.append({"clip_id": clip_id, "reason": "excluded_id_still_canonical"})

    visible_rates: list[float] = []
    two_hand_rates: list[float] = []
    total_frames = 0
    zero_hand_ids: list[str] = []
    old_path_hits: list[str] = []
    for clip_id, json_path in keypoints.items():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        summary = payload.get("summary")
        frames = payload.get("frames")
        if not isinstance(summary, dict) or summary.get("clip_id") != clip_id:
            errors.append({"clip_id": clip_id, "reason": "summary_id_mismatch"})
            continue
        if args.old_root_text in json.dumps(summary, ensure_ascii=False):
            old_path_hits.append(clip_id)
        if not isinstance(frames, list) or not frames:
            errors.append({"clip_id": clip_id, "reason": "empty_frames"})
            continue
        total_frames += len(frames)
        has_hand = False
        previous = -math.inf
        for frame in frames:
            timestamp = frame.get("time_sec")
            hands = frame.get("hands")
            if not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)) or float(timestamp) < previous:
                errors.append({"clip_id": clip_id, "reason": "invalid_timeline"})
                break
            previous = float(timestamp)
            if not isinstance(hands, list) or frame.get("hand_count") != len(hands):
                errors.append({"clip_id": clip_id, "reason": "hand_count_mismatch"})
                break
            has_hand = has_hand or bool(hands)
            for hand in hands:
                points = hand.get("points") if isinstance(hand, dict) else None
                if (
                    not isinstance(points, list)
                    or len(points) != 21
                    or any(
                        not isinstance(point, list)
                        or len(point) != 3
                        or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in point)
                        for point in points
                    )
                ):
                    errors.append({"clip_id": clip_id, "reason": "invalid_21x3_points"})
                    break
        if not has_hand:
            zero_hand_ids.append(clip_id)
        visible_rates.append(float(summary.get("hand_visible_rate") or 0.0))
        two_hand_rates.append(float(summary.get("two_hand_rate") or 0.0))
    if zero_hand_ids:
        errors.append({"reason": "zero_hand_samples_remain", "count": str(len(zero_hand_ids))})
    if old_path_hits:
        errors.append({"reason": "old_root_paths_remain", "count": str(len(old_path_hits))})

    report = {
        "schema_version": 1,
        "status": "video_generation_dataset_ready" if not errors else "failed",
        "dataset_root": str(root),
        "manifest_count": len(rows),
        "clip_count": len(clips),
        "keypoint_json_count": len(keypoints),
        "source_coverage_count": len({str(row.get("source_sha256") or "") for row in rows}),
        "total_clip_size_bytes": sum(path.stat().st_size for path in clips.values()),
        "total_keypoint_size_bytes": sum(path.stat().st_size for path in keypoints.values()),
        "total_frames_sampled": total_frames,
        "hand_visible_ge_30_count": sum(value >= 0.30 for value in visible_rates),
        "hand_visible_ge_50_count": sum(value >= 0.50 for value in visible_rates),
        "hand_visible_ge_70_count": sum(value >= 0.70 for value in visible_rates),
        "two_hand_ge_50_count": sum(value >= 0.50 for value in two_hand_rates),
        "zero_hand_count": len(zero_hand_ids),
        "zero_hand_clip_ids": zero_hand_ids,
        "excluded_zero_hand_ids": args.excluded_id,
        "old_root_path_hit_count": len(old_path_hits),
        "landmarks_per_hand": 21,
        "coordinates_per_landmark": 3,
        "manifest_sha256": sha256(args.manifest),
        "hand_detection_run": True,
        "training_run": False,
        "error_count": len(errors),
        "errors": errors,
    }
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
