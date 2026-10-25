#!/usr/bin/env python3
"""PyTorch 2.6+ compatibility entrypoint for official SCAIL-2 Base conversion."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("module"), dict):
        raise ValueError(f"No module state dict in {path}")
    return checkpoint["module"]


def load_official_converter(path: Path):
    spec = importlib.util.spec_from_file_location("official_scail2_convert_base", path)
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
    args = parser.parse_args()

    actual_sha256 = sha256(args.input)
    if actual_sha256.lower() != args.expected_sha256.lower():
        raise ValueError(
            f"refusing trusted pickle load: SHA256 mismatch; expected "
            f"{args.expected_sha256}, got {actual_sha256}"
        )

    official = load_official_converter(args.official_converter)
    state_dict = load_module_state(args.input)
    mapped = official.get_new_state_dict(state_dict)
    # The official QKV mapping returns views sharing one backing storage.
    # Safetensors requires independent storage, so detach and clone once.
    converted = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in mapped.items()
    }
    del mapped, state_dict
    gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(args.output))
    print(f"Saved {len(converted)} tensors to {args.output}")


if __name__ == "__main__":
    main()
