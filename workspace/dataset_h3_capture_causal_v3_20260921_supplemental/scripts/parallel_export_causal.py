#!/usr/bin/env python3
"""Export a prepared causal manifest in parallel, grouped by raw source."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

if __package__:
    from .causal_clip_pipeline import export_rows, load_jsonl, write_jsonl
else:
    from causal_clip_pipeline import export_rows, load_jsonl, write_jsonl


def export_parallel(
    rows: list[dict[str, Any]], work_root: Path, source_root: Path,
    workers: int,
    exporter: Callable[..., list[dict[str, Any]]] = export_rows,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (str(row.get("source_rel") or row.get("source_video")), str(row.get("source_sha256")))
        grouped[key].append((index, row))
    results: list[dict[str, Any] | None] = [None] * len(rows)

    def run_group(indexed: list[tuple[int, dict[str, Any]]]):
        exported = exporter(
            [row for _, row in indexed], work_root,
            source_root=source_root, write_manifest=False,
        )
        if len(exported) != len(indexed):
            raise RuntimeError("exporter returned wrong row count")
        return [(original_index, result) for (original_index, _), result in zip(indexed, exported)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(run_group, indexed) for indexed in grouped.values()]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            for index, row in future.result():
                results[index] = row
            print(f"[{completed}/{len(futures)}] source groups", flush=True)
    if any(row is None for row in results):
        raise RuntimeError("parallel export left missing rows")
    return [row for row in results if row is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    work_root = Path(args.work_root)
    rows = load_jsonl(Path(args.manifest))
    exported = export_parallel(rows, work_root, Path(args.source_root), args.workers)
    write_jsonl(work_root / "manifests" / "export_manifest.jsonl", exported)
    report = {
        "manifest_count": len(rows),
        "exported_count": sum(row.get("export_status") == "ok" for row in exported),
        "error_count": sum(row.get("export_status") != "ok" for row in exported),
        "generated_count": sum(row.get("cache_status") == "generated" for row in exported),
        "reused_count": sum(row.get("cache_status") == "reused" for row in exported),
        "workers": args.workers,
    }
    report_path = work_root / "reports" / "parallel_export.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
