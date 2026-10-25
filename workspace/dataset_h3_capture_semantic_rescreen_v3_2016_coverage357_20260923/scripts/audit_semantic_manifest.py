#!/usr/bin/env python3
"""Audit semantic candidates and optionally select an ordered smoke manifest."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EPSILON = 1e-6


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("clip_id") or "") for row in rows]
    windows: dict[tuple[str, float, float], list[str]] = collections.defaultdict(list)
    cores: dict[tuple[str, float, float], list[str]] = collections.defaultdict(list)
    contract_errors: list[dict[str, str]] = []
    for row in rows:
        clip_id = str(row.get("clip_id") or "")
        source_sha = str(row.get("source_sha256") or "")
        try:
            start = float(row["source_start_sec"])
            end = float(row["source_end_sec"])
            core_start = float(row["core_start_sec_abs"])
            core_end = float(row["core_end_sec_abs"])
            duration = end - start
            phases = [float(row[field]) for field in PHASE_FIELDS]
            if not all(math.isfinite(value) for value in (start, end, core_start, core_end, *phases)):
                raise ValueError("non-finite numeric field")
            if not 2.0 - EPSILON <= duration <= 20.0 + EPSILON:
                contract_errors.append({"clip_id": clip_id, "reason": "duration_outside_2_to_20"})
            if not start - EPSILON <= core_start < core_end <= end + EPSILON:
                contract_errors.append({"clip_id": clip_id, "reason": "core_outside_export_window"})
            if not all(-EPSILON <= value <= duration + EPSILON for value in phases):
                contract_errors.append({"clip_id": clip_id, "reason": "phase_outside_export_window"})
            if any(left > right + EPSILON for left, right in zip(phases, phases[1:])):
                contract_errors.append({"clip_id": clip_id, "reason": "phase_order_invalid"})
        except (KeyError, TypeError, ValueError):
            contract_errors.append({"clip_id": clip_id, "reason": "missing_or_invalid_contract_field"})
            start = float(row.get("source_start_sec") or 0.0)
            end = float(row.get("source_end_sec") or 0.0)
        windows[
            (
                source_sha,
                round(start, 6),
                round(end, 6),
            )
        ].append(str(row["clip_id"]))
        cores[
            (
                source_sha,
                round(float(row["core_start_sec_abs"]), 6),
                round(float(row["core_end_sec_abs"]), 6),
            )
        ].append(str(row["clip_id"]))

    duplicate_windows = {key: value for key, value in windows.items() if len(value) > 1}
    duplicate_cores = {key: value for key, value in cores.items() if len(value) > 1}
    durations = [float(row["source_end_sec"]) - float(row["source_start_sec"]) for row in rows]
    return {
        "row_count": len(rows),
        "unique_id_count": len(set(ids)),
        "empty_id_count": sum(not value for value in ids),
        "source_coverage_count": len({str(row.get("source_sha256") or "") for row in rows}),
        "duration_total_sec": round(sum(durations), 6),
        "duration_min_sec": min(durations, default=0.0),
        "duration_max_sec": max(durations, default=0.0),
        "unique_export_window_count": len(windows),
        "duplicate_export_window_group_count": len(duplicate_windows),
        "duplicate_export_window_row_count": sum(len(value) for value in duplicate_windows.values()),
        "duplicate_export_window_examples": [
            {"key": list(key), "clip_ids": value}
            for key, value in list(duplicate_windows.items())[:10]
        ],
        "duplicate_core_group_count": len(duplicate_cores),
        "duplicate_core_row_count": sum(len(value) for value in duplicate_cores.values()),
        "invalid_source_sha256_count": sum(
            SHA256_RE.fullmatch(str(row.get("source_sha256") or "")) is None for row in rows
        ),
        "contract_error_count": len(contract_errors),
        "contract_error_examples": contract_errors[:20],
        "pending_user_review_false_count": sum(
            row.get("pending_user_review") is not True for row in rows
        ),
        "semantic_feedback_verified_nonfalse_count": sum(
            row.get("semantic_feedback_verified") is not False for row in rows
        ),
        "navigation_only_count": sum(row.get("navigation_only") is True for row in rows),
        "export_ineligible_count": sum(
            row.get("exportable") is not True or row.get("export_eligible") is not True
            for row in rows
        ),
        "action_family_counts": dict(
            sorted(collections.Counter(str(row.get("action_family") or "missing") for row in rows).items())
        ),
        "proposal_origin_counts": dict(
            sorted(collections.Counter(str(row.get("proposal_origin") or "missing") for row in rows).items())
        ),
    }


def write_selected(
    rows: list[dict[str, Any]], ids_path: Path, output_path: Path
) -> dict[str, Any]:
    requested = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row.get("clip_id")): row for row in rows}
    missing = [clip_id for clip_id in requested if clip_id not in by_id]
    if missing:
        raise ValueError(f"missing requested clip IDs: {missing}")
    selected = [by_id[clip_id] for clip_id in requested]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    return {
        "requested_count": len(requested),
        "selected_count": len(selected),
        "selected_ids": requested,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = load_jsonl(args.manifest)
    result: dict[str, Any] = {"audit": audit(rows)}
    if bool(args.ids) != bool(args.output):
        parser.error("--ids and --output must be provided together")
    if args.ids and args.output:
        result["selection"] = write_selected(rows, args.ids, args.output)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
