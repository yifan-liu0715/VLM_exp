#!/usr/bin/env python3
"""Materialize reviewed semantic additions and exclusions from a JSON policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


COPY_FIELDS = (
    "action", "action_label", "target", "target_type", "target_id", "target_instance_id",
    "direction", "action_direction", "pre_state", "post_state", "object_state_before",
    "object_state_after", "cause_description", "response_description", "camera_motion",
    "visual_evidence", "notes", "confidence",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_decision(parent: dict[str, Any], spec: dict[str, Any], v3_path: Path, row_number: int) -> dict[str, Any]:
    source_start = float(parent["source_start_sec"])
    source_end = float(parent["source_end_sec"])
    action_start_rel = float(parent["action_start_sec_rel"])
    action_end_rel = float(parent["action_end_sec_rel"])
    core_start = round(source_start + action_start_rel, 6)
    core_end = round(source_start + action_end_rel, 6)
    if not source_start <= core_start < core_end <= source_end:
        raise ValueError(f"{parent['clip_id']}: action core falls outside source window")
    decision_id = f"rescreen_v2_{parent['clip_id']}"
    row: dict[str, Any] = {
        "decision_id": decision_id,
        "clip_id": decision_id,
        "source_rel": parent["source_rel"],
        "source_name": parent.get("source_name") or Path(parent["source_rel"]).name,
        "source_sha256": parent["source_sha256"],
        "source_duration_sec": float(parent["source_duration_sec"]),
        "source_start_sec": source_start,
        "source_end_sec": source_end,
        "duration_sec": round(source_end - source_start, 6),
        "core_start_sec": core_start,
        "core_end_sec": core_end,
        "core_start_sec_abs": core_start,
        "core_end_sec_abs": core_end,
        "pre_start_sec_abs": source_start,
        "post_end_sec_abs": source_end,
        "action_family": spec["action_family"],
        "review_status": "pending_user_review",
        "pending_user_review": True,
        "semantic_feedback_verified": False,
        "new_visual_review_performed": True,
        "needs_visual_review": True,
        "export_eligible": True,
        "navigation_only": False,
        "counting_eligible": False,
        "eligibility_reason": "export_for_user_review_only_not_training_approved",
        "manual_presence_override": True,
        "manual_presence_override_reason": spec["manual_presence_override_reason"],
        "core_boundary_method": "three_fps_contact_sheet_frame_review",
        "core_boundary_status": "frame_reviewed_provisional_pending_user_review",
        "rescreen_batch_id": "semantic_rescreen_v2_20260923",
        "rescreen_parent_clip_id": parent["clip_id"],
        "visual_review_artifact": spec["visual_review_artifact"],
        "evidence_refs": [{
            "path": v3_path.as_posix(),
            "row_number": row_number,
            "row_id": parent["clip_id"],
        }],
        "replaces_clip_ids": [],
        "supersedes_clip_ids": [],
        "pre_start_sec_rel": 0.0,
        "pre_end_sec_rel": action_start_rel,
        "action_start_sec_rel": action_start_rel,
        "action_end_sec_rel": action_end_rel,
        "response_start_sec_rel": float(parent.get("response_start_sec_rel", action_end_rel)),
        "response_end_sec_rel": float(parent.get("response_end_sec_rel", action_end_rel)),
        "post_start_sec_rel": float(parent.get("post_start_sec_rel", action_end_rel)),
        "post_end_sec_rel": round(source_end - source_start, 6),
    }
    row.update({field: parent[field] for field in COPY_FIELDS if field in parent})
    row["action_family"] = spec["action_family"]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--existing-decisions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-decisions", type=Path, required=True)
    parser.add_argument("--output-exclude-ids", type=Path, required=True)
    parser.add_argument("--output-changes", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    v3_rows = load_jsonl(args.v3)
    existing = load_jsonl(args.existing_decisions)
    policy = json.loads(args.policy.read_text(encoding="utf-8-sig"))
    v3_index: dict[str, tuple[int, dict[str, Any]]] = {}
    for row_number, row in enumerate(v3_rows, 1):
        clip_id = str(row.get("clip_id") or "")
        if not clip_id or clip_id in v3_index:
            if clip_id:
                raise ValueError(f"duplicate V3 clip_id: {clip_id}")
            continue
        v3_index[clip_id] = (row_number, row)

    additions: list[dict[str, Any]] = []
    for spec in policy["additions"]:
        parent_id = str(spec["parent_clip_id"])
        if parent_id not in v3_index:
            raise ValueError(f"missing V3 parent: {parent_id}")
        row_number, parent = v3_index[parent_id]
        additions.append(build_decision(parent, spec, args.v3, row_number))

    decision_ids = [str(row.get("decision_id") or row.get("clip_id") or "") for row in [*existing, *additions]]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("combined decision ids are not unique")
    exclusion_rows = policy["exclusions"]
    exclusion_ids = [str(row["candidate_id"]) for row in exclusion_rows]
    if len(exclusion_ids) != len(set(exclusion_ids)):
        raise ValueError("exclusion candidate ids are not unique")

    write_jsonl(args.output_decisions, [*existing, *additions])
    args.output_exclude_ids.parent.mkdir(parents=True, exist_ok=True)
    args.output_exclude_ids.write_text("".join(f"{candidate_id}\n" for candidate_id in exclusion_ids), encoding="utf-8")
    changes = [
        {"change": "add", "decision_id": row["decision_id"], "parent_clip_id": row["rescreen_parent_clip_id"],
         "source_rel": row["source_rel"], "source_start_sec": row["source_start_sec"],
         "source_end_sec": row["source_end_sec"], "action_family": row["action_family"],
         "action_label": row.get("action_label"), "review_status": row["review_status"]}
        for row in additions
    ] + [{"change": "exclude", **row} for row in exclusion_rows]
    write_jsonl(args.output_changes, changes)
    report = {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "existing_decision_count": len(existing),
        "addition_count": len(additions),
        "combined_decision_count": len(existing) + len(additions),
        "exclusion_count": len(exclusion_ids),
        "net_requested_change": len(additions) - len(exclusion_ids),
        "all_additions_pending_user_review": all(row["review_status"] == "pending_user_review" for row in additions),
        "all_additions_unverified": all(row["semantic_feedback_verified"] is False for row in additions),
        "manual_presence_override_count": sum(row["manual_presence_override"] is True for row in additions),
        "hand_keypoint_run": False,
        "training_run": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
