#!/usr/bin/env python3
"""Convert original Wan2.2 DiT shards to the exact Diffusers key layout used by EffectV2V."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from models.wan2.transformer_wan import WanTransformer3DModel
from v2v_components import convert_original_wan_shard


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DEFAULT_SOURCE = PROJECT_ROOT / "pretrain_weights/Wan2.2-TI2V-5B-HF"
DEFAULT_OUTPUT = PROJECT_ROOT / "v2v_formal_0715/base_components/transformer"
DEFAULT_CONFIG = PROJECT_ROOT / "v2v_formal_0715/configs/transformer_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_model_keys(config_path: Path) -> set[str]:
    config = WanTransformer3DModel.load_config(config_path)
    with torch.device("meta"):
        model = WanTransformer3DModel.from_config(config)
    return set(model.state_dict())


def convert(source: Path, output: Path, config_path: Path) -> dict:
    source_index_path = source / "diffusion_pytorch_model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    source_weight_map = source_index["weight_map"]
    shard_names = sorted(set(source_weight_map.values()))
    output.mkdir(parents=True, exist_ok=True)

    output_weight_map: dict[str, str] = {}
    shard_records = []
    for shard_name in shard_names:
        source_path = source / shard_name
        output_path = output / shard_name
        temp_path = output / f".{shard_name}.tmp"
        print(f"[convert] {source_path} -> {output_path}", flush=True)
        original = load_file(str(source_path), device="cpu")
        converted = convert_original_wan_shard(original)
        source_keys = {key for key, value in source_weight_map.items() if value == shard_name}
        if len(converted) != len(source_keys):
            raise RuntimeError(
                f"{shard_name}: converted {len(converted)} keys, expected {len(source_keys)}"
            )
        save_file(converted, str(temp_path))
        os.replace(temp_path, output_path)
        for key in converted:
            if key in output_weight_map:
                raise RuntimeError(f"duplicate converted key: {key}")
            output_weight_map[key] = shard_name
        shard_records.append(
            {
                "source": str(source_path),
                "output": str(output_path),
                "source_bytes": source_path.stat().st_size,
                "output_bytes": output_path.stat().st_size,
                "tensor_count": len(converted),
            }
        )
        del original, converted

    expected = expected_model_keys(config_path)
    actual = set(output_weight_map)
    if actual != expected:
        raise RuntimeError(
            f"converted key coverage mismatch: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )

    output_index = {
        "metadata": {"total_size": sum((output / name).stat().st_size for name in shard_names)},
        "weight_map": output_weight_map,
    }
    index_temp = output / ".diffusion_pytorch_model.safetensors.index.json.tmp"
    index_path = output / "diffusion_pytorch_model.safetensors.index.json"
    index_temp.write_text(json.dumps(output_index, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(index_temp, index_path)
    shutil.copy2(config_path, output / "config.json")

    manifest = {
        "source_directory": str(source),
        "source_index": str(source_index_path),
        "source_index_sha256": sha256(source_index_path),
        "output_directory": str(output),
        "config_sha256": sha256(config_path),
        "tensor_count": len(actual),
        "key_coverage": "100%",
        "shards": shard_records,
    }
    manifest_path = output / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = convert(args.source, args.output, args.config)
    print(json.dumps(report, indent=2), flush=True)
