#!/usr/bin/env python3
"""Select an idle physical GPU while enforcing the permanent GPU0 exclusion."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable, Mapping


def select_gpu(rows: Iterable[Mapping[str, int]], minimum_free_mib: int) -> int:
    if minimum_free_mib < 0:
        raise ValueError("minimum_free_mib must be nonnegative")
    eligible = []
    for raw in rows:
        row = {key: int(value) for key, value in raw.items()}
        if row["index"] == 0:
            continue
        if row["utilization_gpu"] > 10:
            continue
        if row["memory_free_mib"] < minimum_free_mib:
            continue
        eligible.append(row)
    if not eligible:
        raise RuntimeError(
            f"no eligible nonzero GPU with at least {minimum_free_mib} MiB free "
            "and utilization <= 10%"
        )
    selected = max(eligible, key=lambda row: (row["memory_free_mib"], -row["index"]))
    return int(selected["index"])


def query_gpus() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "memory_free_mib": int(fields[1]),
                "utilization_gpu": int(fields[2]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-free-mib", type=int, default=16000)
    args = parser.parse_args()
    rows = query_gpus()
    selected = select_gpu(rows, args.minimum_free_mib)
    selected_row = next(row for row in rows if row["index"] == selected)
    print(
        "selected physical GPU "
        f"{selected}: free={selected_row['memory_free_mib']}MiB, "
        f"util={selected_row['utilization_gpu']}%",
        file=sys.stderr,
    )
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
