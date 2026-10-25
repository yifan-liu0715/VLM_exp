#!/usr/bin/env python3
"""Run the existing hand detector in isolated parallel shards."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def shard_ranges(total: int, workers: int) -> list[tuple[int, int]]:
    if total < 0 or workers <= 0:
        raise ValueError("invalid total or workers")
    workers = min(workers, total) if total else 0
    return [(total * i // workers, total * (i + 1) // workers) for i in range(workers)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-fps", type=float, default=24.0)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    total = sum(bool(line.strip()) for line in manifest.read_text(encoding="utf-8-sig").splitlines())
    root = Path(args.shard_root)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, (start, end) in enumerate(shard_ranges(total, args.workers)):
        output = root / f"shard_{index:02d}"
        output.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"shard_{index:02d}.log"
        log = log_path.open("w", encoding="utf-8")
        command = [
            args.python, args.detector, "--manifest", str(manifest),
            "--clip-dir", args.clip_dir, "--out", str(output), "--model", args.model,
            "--sample-fps", str(args.sample_fps), "--start", str(start), "--end", str(end),
            "--opencv-threads", "1",
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((index, start, end, process, log, log_path, output))
    records = []
    for index, start, end, process, log, log_path, output in jobs:
        returncode = process.wait()
        log.close()
        records.append({
            "shard": index, "start": start, "end": end, "count": end - start,
            "returncode": returncode, "log": str(log_path), "output": str(output),
        })
    report = {
        "manifest_count": total, "worker_count": len(records),
        "success_count": sum(row["returncode"] == 0 for row in records),
        "failure_count": sum(row["returncode"] != 0 for row in records),
        "shards": records,
    }
    (root / "sharded_detection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("manifest_count", "worker_count", "success_count", "failure_count")}))
    return 0 if report["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
