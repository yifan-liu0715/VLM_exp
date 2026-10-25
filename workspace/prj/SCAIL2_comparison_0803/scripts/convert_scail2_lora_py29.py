#!/usr/bin/env python3
"""PyTorch 2.6+ compatibility entrypoint for official SCAIL-2 LoRA conversion.

The official converter was written before ``torch.load`` changed its default
``weights_only`` value. This wrapper keeps the official key conversion code and
only makes the trusted-checkpoint loading choice explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for name in ("module", "state_dict", "model"):
            state_dict = checkpoint.get(name)
            if isinstance(state_dict, dict):
                return state_dict
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Could not find a tensor state dict in {path}")


def load_official_converter(path: Path):
    spec = importlib.util.spec_from_file_location("official_scail2_convert_lora", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-converter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--dtype", default="keep")
    parser.add_argument("--print-sample", type=int, default=8)
    args = parser.parse_args()

    actual_sha256 = sha256(args.input)
    if actual_sha256.lower() != args.expected_sha256.lower():
        raise ValueError(
            f"refusing trusted pickle load: SHA256 mismatch; expected "
            f"{args.expected_sha256}, got {actual_sha256}"
        )

    official = load_official_converter(args.official_converter)
    state_dict = load_module_state(args.input)
    converted = official.convert_state_dict(
        state_dict,
        dtype=official._parse_dtype(args.dtype),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(args.output))
    print(f"Saved {len(converted)} tensors to {args.output}")
    for key in list(converted)[: args.print_sample]:
        tensor = converted[key]
        print(f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
