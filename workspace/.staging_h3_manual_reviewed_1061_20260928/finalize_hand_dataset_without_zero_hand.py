#!/usr/bin/env python3
"""Remove audited zero-hand clips and rename a hand-keypoint dataset safely."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8"),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def replace_root(value: Any, old_root: str, new_root: str) -> Any:
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    if isinstance(value, list):
        return [replace_root(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_root(item, old_root, new_root)
            for key, item in value.items()
        }
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--expected-input", type=int, required=True)
    parser.add_argument("--expected-removed", type=int, required=True)
    args = parser.parse_args()

    old_root = args.old_root.resolve()
    new_root = args.new_root.absolute()
    require(old_root.is_dir(), f"missing old root: {old_root}")
    require(not new_root.exists(), f"new root already exists: {new_root}")
    require(old_root.parent == new_root.parent, "old and new roots must share a parent")

    audit_path = old_root / "reports/final_hand_keypoint_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    zero_ids = [str(value) for value in audit.get("zero_hand_clip_ids", [])]
    require(audit.get("status") == "hand_keypoints_complete", "input audit is not complete")
    require(audit.get("clip_count") == args.expected_input, "input audit count mismatch")
    require(len(zero_ids) == args.expected_removed, "zero-hand count mismatch")
    require(len(set(zero_ids)) == len(zero_ids), "duplicate zero-hand IDs")

    manual_path = old_root / "manifests/manual_reviewed_1061.jsonl"
    detection_path = old_root / "manifests/hand_detection_manifest_1061.jsonl"
    manual_rows = load_jsonl(manual_path)
    detection_rows = load_jsonl(detection_path)
    manual_ids = {str(row["clip_id"]) for row in manual_rows}
    detection_ids = {str(row["clip_id"]) for row in detection_rows}
    require(len(manual_rows) == len(manual_ids) == args.expected_input, "manual manifest mismatch")
    require(len(detection_rows) == len(detection_ids) == args.expected_input, "detection manifest mismatch")
    require(manual_ids == detection_ids, "input manifest ID sets differ")
    require(set(zero_ids).issubset(manual_ids), "zero-hand ID absent from manifests")

    for clip_id in zero_ids:
        clip = old_root / "clips" / f"{clip_id}.mp4"
        keypoint = old_root / "hand_keypoints" / f"{clip_id}_hands.json"
        require(clip.is_file(), f"missing zero-hand clip: {clip_id}")
        require(keypoint.is_file(), f"missing zero-hand JSON: {clip_id}")
        payload = json.loads(keypoint.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        require(isinstance(frames, list) and frames, f"invalid zero-hand JSON: {clip_id}")
        require(
            all(int(frame.get("hand_count") or 0) == 0 for frame in frames),
            f"audited zero-hand JSON now contains a hand: {clip_id}",
        )

    os.rename(old_root, new_root)
    old_text = str(old_root)
    new_text = str(new_root.resolve())
    archive = new_root / "archive/pre_zero_hand_filter_1061"
    (archive / "manifests").mkdir(parents=True, exist_ok=True)
    (archive / "reports").mkdir(parents=True, exist_ok=True)
    shutil.copy2(new_root / "manifests/manual_reviewed_1061.jsonl", archive / "manifests/")
    shutil.copy2(new_root / "manifests/hand_detection_manifest_1061.jsonl", archive / "manifests/")
    for name in (
        "final_hand_keypoint_audit.json",
        "keypoint_validation.json",
        "materialization.json",
        "keypoint_merge.json",
        "provenance_sha256.txt",
        "disk_after_detection.txt",
    ):
        source = new_root / "reports" / name
        if source.is_file():
            shutil.copy2(source, archive / "reports" / name)

    zero_set = set(zero_ids)
    kept_manual = [
        replace_root(row, old_text, new_text)
        for row in manual_rows
        if str(row["clip_id"]) not in zero_set
    ]
    kept_detection = [
        replace_root(row, old_text, new_text)
        for row in detection_rows
        if str(row["clip_id"]) not in zero_set
    ]
    expected_output = args.expected_input - args.expected_removed
    require(len(kept_manual) == len(kept_detection) == expected_output, "output count mismatch")
    write_jsonl(new_root / "manifests/manual_reviewed_1058.jsonl", kept_manual)
    write_jsonl(new_root / "manifests/hand_detection_manifest_1058.jsonl", kept_detection)

    for path in (
        new_root / "manifests/manual_reviewed_1061.jsonl",
        new_root / "manifests/hand_detection_manifest_1061.jsonl",
    ):
        path.unlink()

    for row in kept_detection:
        clip_id = str(row["clip_id"])
        json_path = new_root / "hand_keypoints" / f"{clip_id}_hands.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        updated = replace_root(payload, old_text, new_text)
        atomic_write(
            json_path,
            json.dumps(updated, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )

    removed_paths: list[str] = []
    for clip_id in zero_ids:
        candidates = [
            new_root / "clips" / f"{clip_id}.mp4",
            new_root / "hand_keypoints" / f"{clip_id}_hands.json",
            new_root / "review_zero_hand_3" / f"{clip_id}.mp4",
        ]
        candidates.extend(
            (new_root / "hand_keypoints_shards").glob(
                f"shard_*/{clip_id}_hands.json"
            )
        )
        for path in candidates:
            if path.exists():
                require(path.is_file() and not path.is_symlink(), f"unsafe removal target: {path}")
                path.unlink()
                removed_paths.append(str(path))
    review_dir = new_root / "review_zero_hand_3"
    if review_dir.is_dir() and not any(review_dir.iterdir()):
        review_dir.rmdir()

    report = {
        "status": "zero_hand_samples_removed",
        "old_root": old_text,
        "new_root": new_text,
        "input_count": args.expected_input,
        "removed_count": len(zero_ids),
        "output_count": expected_output,
        "removed_clip_ids": zero_ids,
        "removed_path_count": len(removed_paths),
        "removed_paths": removed_paths,
        "archive_root": str(archive),
        "training_run": False,
    }
    write_json(new_root / "reports/zero_hand_filter.json", report)
    atomic_write(
        new_root / "READY",
        b"hand_keypoints_filtered_1058_validation_pending_no_training\n",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
