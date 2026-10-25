#!/usr/bin/env python3
"""Audit a safetensors header without loading tensor payloads."""

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path


def audit(path: Path) -> dict:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    dtypes = Counter(value["dtype"] for value in tensors.values())
    total_numel = sum(math.prod(value["shape"]) for value in tensors.values())
    payload_bytes = max((value["data_offsets"][1] for value in tensors.values()), default=0)
    expected_size = 8 + header_size + payload_bytes
    return {
        "status": "passed" if expected_size == path.stat().st_size else "size_mismatch",
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "header_size_bytes": header_size,
        "tensor_count": len(tensors),
        "total_numel": total_numel,
        "dtypes": dict(sorted(dtypes.items())),
        "payload_bytes": payload_bytes,
        "expected_file_size_bytes": expected_size,
        "sample_tensors": [
            {"key": key, "dtype": value["dtype"], "shape": value["shape"]}
            for key, value in list(tensors.items())[:16]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    if result["status"] != "passed":
        raise RuntimeError(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "tensor_count", "total_numel", "dtypes")}))


if __name__ == "__main__":
    main()
