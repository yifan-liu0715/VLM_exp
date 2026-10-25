#!/usr/bin/env python3
"""Run the isolated native-resolution Scheme 3 V2V LoRA experiment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys

import pytorch_lightning as lightning
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch

from build_teacher_aligned_manifests import hash_tree
from teacher_aligned_dataset import TeacherAlignedPairedDataset
from teacher_aligned_system import TeacherAlignedV2VSystem
from training_contracts import (
    explicit_nonzero_visible_gpus,
    validate_training_config,
)


class LatestAdapterCheckpoint(Callback):
    def __init__(self, output_root: Path, every_n_steps: int):
        self.output_file = output_root / "checkpoints" / "latest_adapter.safetensors"
        self.every_n_steps = int(every_n_steps)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if (
            trainer.is_global_zero
            and trainer.global_step > 0
            and trainer.global_step % self.every_n_steps == 0
        ):
            metadata = pl_module.export_trainable_weights(self.output_file)
            print(f"[checkpoint] compact latest: {metadata}", flush=True)


class OverwriteRecoveryCheckpoint(Callback):
    """Atomically overwrite one resumable checkpoint at a fixed interval."""

    def __init__(self, output_root: Path, every_n_steps: int):
        self.output_file = output_root / "checkpoints" / "recovery.ckpt"
        self.every_n_steps = int(every_n_steps)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if (
            trainer.is_global_zero
            and trainer.global_step > 0
            and trainer.global_step % self.every_n_steps == 0
        ):
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_file.with_name(
                f".{self.output_file.name}.tmp"
            )
            trainer.save_checkpoint(str(temporary), weights_only=False)
            os.replace(temporary, self.output_file)
            print(
                f"[checkpoint] resumable recovery: {self.output_file}",
                flush=True,
            )


def environment_report() -> dict:
    versions = {}
    for name in (
        "torch",
        "pytorch_lightning",
        "diffusers",
        "transformers",
        "peft",
        "omegaconf",
        "decord",
        "safetensors",
    ):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            versions[name] = f"ERROR: {error!r}"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_trainer(cfg, visible: list[str], output_root: Path):
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if not bool(cfg.training.get("preflight", False)):
        callbacks.append(
            LatestAdapterCheckpoint(
                output_root,
                int(cfg.training.checkpoint_every_n_steps),
            )
        )
        callbacks.append(
            OverwriteRecoveryCheckpoint(
                output_root,
                int(cfg.training.checkpoint_every_n_steps),
            )
        )
    return lightning.Trainer(
        accelerator="gpu",
        devices=len(visible),
        strategy="ddp" if len(visible) > 1 else "auto",
        precision=str(cfg.training.precision),
        max_steps=int(cfg.training.max_steps),
        max_epochs=int(cfg.training.max_epochs),
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        gradient_clip_val=cfg.training.gradient_clip_val,
        logger=CSVLogger(save_dir=str(output_root / "logs"), name="lightning"),
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        deterministic=bool(cfg.training.deterministic),
        benchmark=not bool(cfg.training.deterministic),
        enable_checkpointing=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    visible = explicit_nonzero_visible_gpus(
        os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    cfg = OmegaConf.load(args.config)
    validate_training_config(OmegaConf.to_container(cfg, resolve=True))

    formal_root = Path(cfg.output.root)
    cfg.training.preflight = False
    output_root = formal_root
    output_root.mkdir(parents=True, exist_ok=True)
    cfg.output.root = str(formal_root)
    OmegaConf.save(cfg, output_root / "resolved_train_config.yaml")
    atomic_json(output_root / "environment.json", environment_report())

    lightning.seed_everything(int(cfg.training.seed), workers=True)
    dataset = TeacherAlignedPairedDataset(
        cfg.data.train_manifest,
        sample_frames=int(cfg.data.num_frames),
        bucket_align=int(cfg.data.bucket_align),
        required_split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.data.num_workers) > 0,
        drop_last=True,
    )
    run_start = {
        "status": "training_started",
        "physical_gpus": visible,
        "train_rows": len(dataset),
        "effective_global_batch": (
            int(cfg.data.batch_size)
            * int(cfg.training.accumulate_grad_batches)
            * len(visible)
            * int(cfg.training.num_nodes)
        ),
        "teacher_reference_sha256": hash_tree(
            Path(__file__).resolve().parents[1] / "teacher_reference"
        ),
        "controlled_exception": (
            "Native DiffSynth VAE/text encoder loading is retained because "
            "the complete Diffusers subfolders are unavailable."
        ),
    }
    atomic_json(output_root / "run_start_manifest.json", run_start)
    system = TeacherAlignedV2VSystem(OmegaConf.to_container(cfg, resolve=True))
    trainer = build_trainer(cfg, visible, output_root)
    recovery_file = formal_root / "checkpoints" / "recovery.ckpt"
    recovery_path = str(recovery_file) if recovery_file.is_file() else None
    if recovery_path:
        print(f"[resume] loading recovery checkpoint: {recovery_path}", flush=True)
    trainer.fit(
        system,
        train_dataloaders=loader,
        ckpt_path=recovery_path,
    )

    if trainer.is_global_zero:
        manifest = {
            **run_start,
            "status": "training_complete",
            "global_step": int(trainer.global_step),
        }
        atomic_json(formal_root / "experiment_manifest.json", manifest)


if __name__ == "__main__":
    main()
