#!/usr/bin/env python3
"""Materialize a reviewed V3 subset from an already validated V2 dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


PHASE_FIELDS = (
    "pre_start_sec_rel",
    "pre_end_sec_rel",
    "action_start_sec_rel",
    "action_end_sec_rel",
    "response_start_sec_rel",
    "response_end_sec_rel",
    "post_start_sec_rel",
    "post_end_sec_rel",
)
MATERIALIZATION_FIELDS = {
    "source_video",
    "max_duration_sec",
    "pipeline_version",
    "boundary_start_sec_text",
    "boundary_end_sec_text",
    "boundary_signature",
    "boundary_signature_algorithm",
    "phase_signature",
    "resolved_source_video",
    "output_video",
    "export_status",
    "cache_status",
    "export_signature",
    "output_sha256",
    "output",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_bytes(path, payload.encode("utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def index_unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        clip_id = str(row.get("clip_id") or row.get("candidate_id") or "")
        require(bool(clip_id), f"{name}: empty clip ID")
        require(clip_id not in indexed, f"{name}: duplicate clip ID {clip_id}")
        indexed[clip_id] = row
    return indexed


def _same_inode(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    return left_stat.st_dev == right_stat.st_dev and left_stat.st_ino == right_stat.st_ino


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )


def _ensure_v3_directory(v3_root: Path, relative: str) -> Path:
    path = v3_root / relative
    require(not _is_link_like(path), f"V3 output directory is a link: {path}")
    path.mkdir(parents=True, exist_ok=True)
    require(path.resolve().is_relative_to(v3_root), f"V3 output directory escapes root: {path}")
    return path


def _copy_identical_or_new(source: Path, target: Path) -> None:
    require(not _is_link_like(target), f"V3 manifest target is a link: {target}")
    if source.resolve() == target.resolve():
        return
    if target.exists():
        require(sha256_file(target) == sha256_file(source), f"existing manifest differs: {target}")
        return
    shutil.copy2(source, target)


def _safe_v2_clip(v2_root: Path, row: dict[str, Any], clip_id: str) -> Path:
    video = Path(str(row.get("output_video") or "")).resolve()
    expected_parent = (v2_root / "clips").resolve()
    require(video.is_file(), f"{clip_id}: missing V2 output video")
    require(video.parent == expected_parent, f"{clip_id}: V2 output path escapes clips directory")
    require(video.name == f"{clip_id}.mp4", f"{clip_id}: unexpected V2 output filename")
    return video


def _validate_semantic_contract(
    candidates: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    parent_ids = {str(value) for value in policy["excluded_parent_ids"]}
    expected_parent_counts = {
        str(key): int(value)
        for key, value in policy["expected_removed_parent_counts"].items()
    }
    actual_parent_counts: collections.Counter[str] = collections.Counter()
    for row in removed:
        hits = parent_ids.intersection(str(value) for value in row.get("v3_parent_ids", []) or [])
        require(len(hits) == 1, f"{row.get('candidate_id')}: expected exactly one excluded parent")
        actual_parent_counts.update(hits)
    require(dict(sorted(actual_parent_counts.items())) == dict(sorted(expected_parent_counts.items())),
            f"removed parent distribution mismatch: {dict(actual_parent_counts)}")

    allowed_families = set(policy["allowed_removed_action_families"])
    allowed_labels = set(policy["allowed_removed_action_labels"])
    require(all(row.get("action_family") in allowed_families for row in removed),
            "removed manifest contains a classified semantic action")
    require(all(row.get("action_label") in allowed_labels for row in removed),
            "removed manifest contains a disallowed action label")
    require(all(not (parent_ids.intersection(str(value) for value in row.get("v3_parent_ids", []) or []))
                for row in candidates), "kept manifest still contains a continuous-event fragment")

    source_count = len({str(row.get("source_sha256") or "") for row in candidates})
    require(source_count == int(policy["expected_output_source_coverage_count"]),
            f"source coverage mismatch: {source_count}")
    require(all(row.get("pending_user_review") is True for row in candidates),
            "a kept candidate is not pending user review")
    require(all(row.get("semantic_feedback_verified") is False for row in candidates),
            "a kept candidate incorrectly claims semantic verification")
    require(all(row.get("exportable") is True and row.get("export_eligible") is True
                for row in candidates), "a kept candidate is not export eligible")

    overrides = [row for row in candidates if row.get("hand_presence_override_applied") is True]
    expected_overrides = int(policy["expected_manual_presence_override_reason_count"])
    require(len(overrides) == expected_overrides, f"manual override count mismatch: {len(overrides)}")
    for row in overrides:
        clip_id = str(row.get("clip_id") or row.get("candidate_id"))
        require(bool(row.get("manual_presence_override_reason")), f"{clip_id}: missing override reason")
        require(bool(row.get("visual_review_artifact")), f"{clip_id}: missing visual review artifact")
        require(bool(row.get("evidence_refs")), f"{clip_id}: missing review evidence")
        require(row.get("core_boundary_method") == "three_fps_contact_sheet_frame_review",
                f"{clip_id}: unexpected boundary review method")
    return {
        "source_coverage_count": source_count,
        "removed_parent_counts": dict(sorted(actual_parent_counts.items())),
        "manual_presence_override_reason_count": len(overrides),
    }


def materialize_dataset(
    *,
    v2_root: Path,
    v3_root: Path,
    candidates_path: Path,
    removed_path: Path,
    policy_path: Path,
    expected_candidate_sha256: str,
    expected_removed_sha256: str,
    expected_policy_sha256: str,
) -> dict[str, Any]:
    v2_root = v2_root.resolve()
    requested_v3_root = v3_root.expanduser().absolute()
    require(not _is_link_like(requested_v3_root), f"V3 root is a link: {requested_v3_root}")
    v3_root = requested_v3_root.resolve()
    candidates_path = candidates_path.resolve()
    removed_path = removed_path.resolve()
    policy_path = policy_path.resolve()
    require(v2_root.is_dir(), f"missing V2 root: {v2_root}")
    require(v2_root != v3_root, "V2 and V3 roots must differ")
    require(sha256_file(candidates_path) == expected_candidate_sha256, "candidate SHA256 mismatch")
    require(sha256_file(removed_path) == expected_removed_sha256, "removed SHA256 mismatch")
    require(sha256_file(policy_path) == expected_policy_sha256, "policy SHA256 mismatch")

    candidates = load_jsonl(candidates_path)
    removed = load_jsonl(removed_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    expected_v2_count = int(policy["expected_input_count"])
    expected_removed_count = int(policy["expected_removed_count"])
    expected_v3_count = int(policy["expected_output_count"])
    require(len(candidates) == expected_v3_count, f"candidate count mismatch: {len(candidates)}")
    require(len(removed) == expected_removed_count, f"removed count mismatch: {len(removed)}")
    for row in candidates:
        injected = sorted(MATERIALIZATION_FIELDS.intersection(row))
        require(not injected, f"{row.get('candidate_id')}: candidate contains materialization field: {injected}")
    semantic_audit = _validate_semantic_contract(candidates, removed, policy)

    v2_candidates = load_jsonl(v2_root / "manifests/semantic_review_candidates_2045.jsonl")
    v2_prepared = load_jsonl(v2_root / "manifests/causal_segments.jsonl")
    v2_exports = load_jsonl(v2_root / "manifests/export_manifest.jsonl")
    require(len(v2_candidates) == len(v2_prepared) == len(v2_exports) == expected_v2_count,
            "V2 manifest counts do not match the policy input count")
    candidate_by_id = index_unique(candidates, "V3 candidates")
    removed_by_id = index_unique(removed, "V3 removed")
    v2_candidate_by_id = index_unique(v2_candidates, "V2 candidates")
    v2_prepared_by_id = index_unique(v2_prepared, "V2 prepared")
    v2_export_by_id = index_unique(v2_exports, "V2 exports")
    kept_ids = set(candidate_by_id)
    removed_ids = set(removed_by_id)
    v2_ids = set(v2_candidate_by_id)
    require(not kept_ids.intersection(removed_ids), "kept and removed ID sets overlap")
    require(kept_ids.union(removed_ids) == v2_ids, "V3 is not an exact keep/remove partition of V2")
    require(set(v2_prepared_by_id) == v2_ids == set(v2_export_by_id), "V2 manifest ID sets differ")

    for clip_id, row in candidate_by_id.items():
        old = v2_candidate_by_id[clip_id]
        for key, old_value in old.items():
            if key in row:
                require(row[key] == old_value, f"{clip_id}: candidate field changed: {key}")

    marker = v3_root / ".semantic_rescreen_v3_marker"
    if v3_root.exists():
        require(v3_root.is_dir(), f"V3 root is not a directory: {v3_root}")
    if v3_root.exists() and any(v3_root.iterdir()):
        require(not _is_link_like(marker), f"V3 marker is a link: {marker}")
        require(marker.is_file(), f"refusing non-empty unmarked V3 root: {v3_root}")
        require(marker.read_text(encoding="utf-8").strip() == str(policy["policy_id"]),
                "existing V3 marker does not match policy")
    keypoint_root = v3_root / "hand_keypoints"
    keypoint_files = sorted(path for path in keypoint_root.rglob("*") if path.is_file()) if keypoint_root.is_dir() else []
    require(not keypoint_files, f"V3 contains hand keypoint artifacts: {keypoint_files[:3]}")
    for relative in ("clips", "state", "state/exports", "manifests", "reports", "scripts", "logs"):
        _ensure_v3_directory(v3_root, relative)
    _atomic_write_bytes(marker, (str(policy["policy_id"]) + "\n").encode("utf-8"))

    _copy_identical_or_new(candidates_path, v3_root / "manifests/semantic_review_candidates_2016.jsonl")
    _copy_identical_or_new(removed_path, v3_root / "manifests/continuous_long_event_removed_29.jsonl")
    _copy_identical_or_new(policy_path, v3_root / "manifests/semantic_continuous_long_event_policy_v3.json")

    prepared_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    total_size = 0
    output_hashes: set[str] = set()
    linked_count = 0
    copied_sidecars = 0
    for clip_id in candidate_by_id:
        candidate = candidate_by_id[clip_id]
        old_prepared = v2_prepared_by_id[clip_id]
        old_export = v2_export_by_id[clip_id]
        require(old_export.get("export_status") == "ok", f"{clip_id}: V2 export status is not ok")
        prepared = dict(old_prepared)
        prepared.update({key: value for key, value in candidate.items() if key not in prepared})
        for field in ("source_rel", "source_sha256", "source_start_sec", "source_end_sec",
                      "duration_sec", "core_start_sec_abs", "core_end_sec_abs", *PHASE_FIELDS):
            require(prepared.get(field) == old_prepared.get(field), f"{clip_id}: changed boundary field {field}")
        require(old_export.get("boundary_signature") == old_prepared.get("boundary_signature"),
                f"{clip_id}: V2 boundary signature mismatch")
        require(old_export.get("phase_signature") == old_prepared.get("phase_signature"),
                f"{clip_id}: V2 phase signature mismatch")

        source_video = _safe_v2_clip(v2_root, old_export, clip_id)
        actual_sha = sha256_file(source_video)
        require(actual_sha == old_export.get("output_sha256"), f"{clip_id}: V2 output SHA mismatch")
        source_sidecar = v2_root / "state/exports" / f"{clip_id}.json"
        require(source_sidecar.is_file(), f"{clip_id}: missing V2 sidecar")
        sidecar_text = source_sidecar.read_text(encoding="utf-8")
        require(str(v2_root) not in sidecar_text, f"{clip_id}: sidecar embeds V2 absolute path")
        sidecar = json.loads(sidecar_text)
        require(sidecar.get("clip_id") == clip_id, f"{clip_id}: sidecar ID mismatch")
        require(sidecar.get("boundary_signature") == old_export.get("boundary_signature"),
                f"{clip_id}: sidecar boundary signature mismatch")
        require(sidecar.get("export_signature") == old_export.get("export_signature"),
                f"{clip_id}: sidecar export signature mismatch")
        require(sidecar.get("output_sha256") == actual_sha, f"{clip_id}: sidecar SHA mismatch")
        require(sidecar.get("output") == old_export.get("output"), f"{clip_id}: sidecar output metadata mismatch")

        target_video = v3_root / "clips" / f"{clip_id}.mp4"
        require(not _is_link_like(target_video), f"{clip_id}: V3 video target is a link")
        if target_video.exists():
            require(_same_inode(source_video, target_video), f"{clip_id}: existing V3 video is not V2 hardlink")
        else:
            os.link(source_video, target_video)
        require(_same_inode(source_video, target_video), f"{clip_id}: hardlink verification failed")
        linked_count += 1

        target_sidecar = v3_root / "state/exports" / f"{clip_id}.json"
        require(not _is_link_like(target_sidecar), f"{clip_id}: V3 sidecar target is a link")
        if target_sidecar.exists():
            require(target_sidecar.read_bytes() == source_sidecar.read_bytes(),
                    f"{clip_id}: existing V3 sidecar differs")
        else:
            shutil.copy2(source_sidecar, target_sidecar)
        require(not _same_inode(source_sidecar, target_sidecar), f"{clip_id}: V3 sidecar must be an independent copy")
        copied_sidecars += 1

        export_row = {
            **old_export,
            **prepared,
            "resolved_source_video": old_export.get("resolved_source_video") or prepared.get("source_video"),
            "output_video": str(target_video.resolve()),
            "export_status": "ok",
            "cache_status": "reused_v2_verified_hardlink",
            "output_sha256": actual_sha,
            "output": sidecar.get("output") or old_export.get("output"),
        }
        require(export_row.get("boundary_signature") == sidecar.get("boundary_signature"),
                f"{clip_id}: final boundary signature differs from sidecar")
        require(export_row.get("export_signature") == sidecar.get("export_signature"),
                f"{clip_id}: final export signature differs from sidecar")
        prepared_rows.append(prepared)
        export_rows.append(export_row)
        total_size += target_video.stat().st_size
        output_hashes.add(actual_sha)

    require(len(output_hashes) == expected_v3_count, "V3 output videos are not content-unique")
    expected_video_names = {f"{clip_id}.mp4" for clip_id in candidate_by_id}
    actual_video_names = {path.name for path in (v3_root / "clips").glob("*.mp4")}
    require(actual_video_names == expected_video_names, "V3 clips directory contains missing or unexpected videos")
    expected_sidecar_names = {f"{clip_id}.json" for clip_id in candidate_by_id}
    actual_sidecar_names = {path.name for path in (v3_root / "state/exports").glob("*.json")}
    require(actual_sidecar_names == expected_sidecar_names, "V3 sidecar directory contains missing or unexpected files")
    write_jsonl(v3_root / "manifests/causal_segments.jsonl", prepared_rows)
    write_jsonl(v3_root / "manifests/export_manifest.jsonl", export_rows)

    expected_review_count = int(policy["expected_review_additions_count"])
    review_count = 0
    target_review = v3_root / "review_additions_14"
    require(not _is_link_like(target_review), f"V3 review additions directory is a link: {target_review}")
    if expected_review_count:
        source_review = v2_root / "review_additions_14"
        review_files = sorted(source_review.glob("*.mp4"))
        require(len(review_files) == expected_review_count, "V2 review additions count mismatch")
        expected_review_names = {path.name for path in review_files}
        existing_review_names = {path.name for path in target_review.glob("*.mp4")} if target_review.is_dir() else set()
        require(existing_review_names.issubset(expected_review_names), "unexpected V3 review addition video")
        override_hashes = {
            row["output_sha256"]
            for row in export_rows
            if row.get("hand_presence_override_applied") is True
        }
        _ensure_v3_directory(v3_root, "review_additions_14")
        seen_review_hashes: set[str] = set()
        for source in review_files:
            digest = sha256_file(source)
            require(digest in override_hashes, f"unexpected V2 review addition: {source.name}")
            require(digest not in seen_review_hashes, f"duplicate V2 review addition: {source.name}")
            target = target_review / source.name
            require(not _is_link_like(target), f"V3 review addition target is a link: {target}")
            if target.exists():
                require(_same_inode(source, target), f"existing review addition is not a hardlink: {source.name}")
            else:
                os.link(source, target)
            seen_review_hashes.add(digest)
            review_count += 1
        require(seen_review_hashes == override_hashes, "review additions do not cover every override")
        require({path.name for path in target_review.glob("*.mp4")} == expected_review_names,
                "V3 review additions directory is incomplete")
    else:
        existing_review_files = sorted(target_review.glob("*.mp4")) if target_review.is_dir() else []
        require(not existing_review_files, "unexpected V3 review addition video")

    report = {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "parent_v2_root": str(v2_root),
        "v3_root": str(v3_root),
        "v2_candidate_count": len(v2_candidates),
        "materialized_count": len(export_rows),
        "removed_count": len(removed),
        "strict_subset_no_additions": kept_ids.issubset(v2_ids),
        "partition_complete": kept_ids.union(removed_ids) == v2_ids,
        **semantic_audit,
        "hardlinked_video_count": linked_count,
        "copied_sidecar_count": copied_sidecars,
        "unique_output_sha256_count": len(output_hashes),
        "total_output_size_bytes": total_size,
        "review_additions_count": review_count,
        "candidate_manifest_sha256": sha256_file(candidates_path),
        "removed_manifest_sha256": sha256_file(removed_path),
        "policy_sha256": sha256_file(policy_path),
        "hand_keypoint_file_count": len(keypoint_files),
        "hand_keypoint_run": False,
        "training_run": False,
    }
    write_json(v3_root / "reports/materialization_report.json", report)
    write_json(v3_root / "reports/provenance.json", {
        "parent_v2_root": str(v2_root),
        "policy_id": policy["policy_id"],
        "selection_rule": policy.get("rule"),
        "disposition": policy.get("disposition"),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--removed", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-removed-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    args = parser.parse_args()
    report = materialize_dataset(
        v2_root=args.v2_root,
        v3_root=args.v3_root,
        candidates_path=args.candidates,
        removed_path=args.removed,
        policy_path=args.policy,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_removed_sha256=args.expected_removed_sha256,
        expected_policy_sha256=args.expected_policy_sha256,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
