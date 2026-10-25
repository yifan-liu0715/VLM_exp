#!/usr/bin/env python3
"""Validate that a capture manifest covers every source in an inventory."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot_read_jsonl:{path}:{exc}") from exc


def source_names(rows: list[dict[str, Any]], label: str) -> tuple[list[str], list[int]]:
    names: list[str] = []
    invalid_rows: list[int] = []
    for line_number, row in enumerate(rows, start=1):
        source_name = row.get("source_name")
        if not isinstance(source_name, str) or not source_name.strip():
            invalid_rows.append(line_number)
            continue
        names.append(source_name)
    return names, invalid_rows


def duplicate_names(names: list[str]) -> list[str]:
    return sorted(name for name, count in Counter(names).items() if count > 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, help="Source inventory JSONL")
    parser.add_argument("--manifest", required=True, help="Capture manifest JSONL")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--expected-source-count", type=int, default=None)
    args = parser.parse_args()

    inventory_rows = load_jsonl(Path(args.inventory))
    manifest_rows = load_jsonl(Path(args.manifest))
    inventory_names, invalid_inventory_rows = source_names(inventory_rows, "inventory")
    manifest_names, invalid_manifest_rows = source_names(manifest_rows, "manifest")
    inventory_set = set(inventory_names)
    manifest_set = set(manifest_names)
    uncovered = sorted(inventory_set - manifest_set)
    unexpected_manifest_sources = sorted(manifest_set - inventory_set)
    expected_source_count_mismatch = (
        args.expected_source_count is not None
        and len(inventory_set) != args.expected_source_count
    )
    report = {
        "inventory": str(Path(args.inventory)),
        "manifest": str(Path(args.manifest)),
        "inventory_source_count": len(inventory_set),
        "manifest_source_count": len(manifest_set),
        "source_count": len(inventory_set),
        "covered": len(inventory_set & manifest_set),
        "covered_source_count": len(inventory_set & manifest_set),
        "uncovered": uncovered,
        "uncovered_sources": uncovered,
        "inventory_only_sources": uncovered,
        "unexpected_manifest_sources": unexpected_manifest_sources,
        "manifest_only_sources": unexpected_manifest_sources,
        "sample_count": len(manifest_rows),
        "expected_source_count": args.expected_source_count,
        "expected_source_count_mismatch": expected_source_count_mismatch,
        "coverage_kind_counts": dict(sorted(
            Counter(str(row.get("coverage_kind") or "base_auto") for row in manifest_rows).items()
        )),
        "duplicate_inventory_sources": duplicate_names(inventory_names),
        "duplicate_manifest_sources": duplicate_names(manifest_names),
        "invalid_inventory_rows": invalid_inventory_rows,
        "invalid_manifest_rows": invalid_manifest_rows,
    }
    report["error_count"] = sum((
        len(report["uncovered"]),
        len(report["unexpected_manifest_sources"]),
        len(report["duplicate_inventory_sources"]),
        len(report["invalid_inventory_rows"]),
        len(report["invalid_manifest_rows"]),
        int(report["expected_source_count_mismatch"]),
    ))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "source_count", "covered", "uncovered", "coverage_kind_counts",
        "expected_source_count", "expected_source_count_mismatch", "error_count",
    )}, ensure_ascii=False))
    return 0 if report["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
