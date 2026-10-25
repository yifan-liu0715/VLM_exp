#!/usr/bin/env python3
"""Audit a SCAIL-2 DPO checkpoint without materializing tensor storage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def walk(value: Any, path: str = ""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from walk(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")


def summarize(path: Path) -> dict:
    checkpoint = torch.load(
        path,
        map_location="meta",
        mmap=True,
        weights_only=False,
    )
    tensors = list(walk(checkpoint))
    total_numel = sum(tensor.numel() for _, tensor in tensors)
    return {
        "status": "passed",
        "checkpoint": str(path),
        "checkpoint_size_bytes": path.stat().st_size,
        "top_level_type": type(checkpoint).__name__,
        "top_level_keys": sorted(map(str, checkpoint.keys())) if isinstance(checkpoint, dict) else [],
        "tensor_count": len(tensors),
        "total_numel": total_numel,
        "dtypes": dict(sorted(Counter(str(t.dtype) for _, t in tensors).items())),
        "all_tensors_meta": all(t.device.type == "meta" for _, t in tensors),
        "tensors": [
            {
                "path": tensor_path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
            for tensor_path, tensor in tensors
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.checkpoint)
    if not result["all_tensors_meta"]:
        raise RuntimeError("checkpoint audit materialized a non-meta tensor")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "tensor_count", "total_numel", "dtypes")}))


if __name__ == "__main__":
    main()
