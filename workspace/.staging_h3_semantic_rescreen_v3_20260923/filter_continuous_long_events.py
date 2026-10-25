#!/usr/bin/env python3
"""Remove unverified fragments derived from reviewed continuous long events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def filter_rows(
    rows: list[dict[str, Any]],
    excluded_parent_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        parent_hits = sorted(excluded_parent_ids.intersection(
            str(value) for value in row.get("v3_parent_ids", []) or []
        ))
        if parent_hits and row.get("action_family") == "interaction_unverified":
            removed.append({
                **row,
                "selected_for_export": False,
                "exportable": False,
                "export_eligible": False,
                "retirement_reason": "continuous_long_event_not_fragmented",
                "excluded_by_parent_ids": parent_hits,
            })
            continue
        if row.get("hand_presence_override_applied") is True and row.get("hand_presence_override_reason"):
            row["manual_presence_override_reason"] = row["hand_presence_override_reason"]
        kept.append(row)
    return kept, removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--removed", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8-sig"))
    parent_ids = {str(value) for value in policy["excluded_parent_ids"]}
    rows = load_jsonl(args.input)
    kept, removed = filter_rows(rows, parent_ids)
    expected_input = int(policy["expected_input_count"])
    expected_removed = int(policy["expected_removed_count"])
    expected_output = int(policy["expected_output_count"])
    if (len(rows), len(removed), len(kept)) != (expected_input, expected_removed, expected_output):
        raise ValueError(
            f"count contract mismatch: input={len(rows)} removed={len(removed)} output={len(kept)}"
        )
    if len({row["candidate_id"] for row in kept}) != len(kept):
        raise ValueError("final candidate IDs are not unique")
    if any(row.get("action_family") != "interaction_unverified" for row in removed):
        raise ValueError("filter attempted to remove a semantic action")
    if {parent for row in removed for parent in row["excluded_by_parent_ids"]} != parent_ids:
        raise ValueError("not every excluded continuous parent matched a selected fragment")

    write_jsonl(args.output, kept)
    write_jsonl(args.removed, removed)
    report = {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "input_count": len(rows),
        "removed_count": len(removed),
        "output_count": len(kept),
        "source_coverage_count": len({row["source_sha256"] for row in kept}),
        "removed_parent_counts": {
            parent: sum(parent in row["excluded_by_parent_ids"] for row in removed)
            for parent in sorted(parent_ids)
        },
        "manual_presence_override_reason_count": sum(
            bool(row.get("manual_presence_override_reason")) for row in kept
        ),
        "input_sha256": sha256_file(args.input),
        "hand_keypoint_run": False,
        "training_run": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
