#!/usr/bin/env python3
"""Capture immutable provenance immediately before the formal training process starts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import accelerate
import diffusers
import peft
import pytorch_lightning
import torch
import transformers
from omegaconf import OmegaConf


ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
FORMAL_ROOT = ROOT / "v2v_formal_0715"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=False, text=True, capture_output=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, default=FORMAL_ROOT / "reports/run_manifest.json")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_report_path = FORMAL_ROOT / "metadata/data_manifest_full_0715.json"
    conversion_report_path = FORMAL_ROOT / "base_components/transformer/conversion_manifest.json"
    tracked_files = sorted(
        list((FORMAL_ROOT / "scripts").rglob("*.py"))
        + list((FORMAL_ROOT / "scripts").rglob("*.sh"))
        + list((FORMAL_ROOT / "configs").glob("*"))
    )
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_pid": os.getppid(),
        "hostname": platform.node(),
        "user": getpass.getuser(),
        "project_root": str(ROOT),
        "formal_root": str(FORMAL_ROOT),
        "command": args.command,
        "physical_gpu_ids": args.gpus.split(","),
        "gpu0_forbidden": True,
        "gpu_snapshot": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines(),
        "git_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "-C", str(ROOT), "status", "--short"]),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "data_manifest": json.loads(data_report_path.read_text(encoding="utf-8")),
        "transformer_conversion": json.loads(conversion_report_path.read_text(encoding="utf-8")),
        "pretrained_input_format": "Wan-AI original checkpoint layout downloaded from Hugging Face",
        "pretrained_input_root": "/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/pretrain_weights/Wan2.2-TI2V-5B-HF",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "pytorch_lightning": pytorch_lightning.__version__,
            "peft": peft.__version__,
        },
        "file_sha256": {str(path): sha256(path) for path in tracked_files if path.is_file()},
        "checkpoint_policy": {
            "latest": str(FORMAL_ROOT / "checkpoints/latest_v2v_lora_and_source_patch.safetensors"),
            "overwrite_interval_steps": int(cfg.training.checkpoint_every_n_steps),
            "contents": "LoRA attention weights + patch_embedding_extra only",
            "optimizer_state_saved": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
