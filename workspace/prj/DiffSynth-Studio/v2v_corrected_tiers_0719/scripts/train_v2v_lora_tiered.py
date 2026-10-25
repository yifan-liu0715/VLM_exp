#!/usr/bin/env python3
"""Corrected-direction Wan2.2 V2V LoRA training with immutable tier milestones."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
FORMAL_SCRIPTS = PROJECT_ROOT / "v2v_formal_0715/scripts"
if str(FORMAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FORMAL_SCRIPTS))

from paired_v2v_dataset import PairedV2VDataset  # noqa: E402
from train_v2v_lora import FormalPairedV2VSystem, LatestAdapterCheckpoint  # noqa: E402
from tiered_training_utils import milestone_adapter_path, validate_milestones  # noqa: E402


class TieredPairedV2VSystem(FormalPairedV2VSystem):
    """Reuse the proven loss/model path and isolate the final export by tier."""

    def on_train_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        cfg = self.cfg
        milestones = {int(step): str(name) for step, name in cfg.training.milestones.items()}
        path = milestone_adapter_path(cfg.output.root, int(cfg.training.max_steps), milestones)
        metadata = self.export_trainable_weights(path)
        print(f"[milestone] ensured final recommended adapter: {metadata}", flush=True)


class TierMilestoneCheckpoint(Callback):
    """Write each configured adapter exactly once when its global step is reached."""

    def __init__(self, output_root: Path, milestones: dict[int, str]):
        self.output_root = output_root
        self.milestones = milestones
        self.saved_steps: set[int] = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if not trainer.is_global_zero or step not in self.milestones or step in self.saved_steps:
            return
        output_file = milestone_adapter_path(self.output_root, step, self.milestones)
        metadata = pl_module.export_trainable_weights(output_file)
        self.saved_steps.add(step)
        record = {
            "tier": self.milestones[step],
            "global_step": step,
            "adapter": str(output_file),
            "metadata": metadata,
        }
        report = self.output_root / self.milestones[step] / "milestone.json"
        report.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[milestone] wrote immutable tier: {record}", flush=True)


def explicit_nonzero_visible_gpus() -> list[str]:
    visible = [gpu.strip() for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.strip()]
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly list nonzero physical GPUs")
    if any(not gpu.isdigit() for gpu in visible) or "0" in visible:
        raise RuntimeError("GPU0 is forbidden for corrected V2V training")
    if len(visible) > 3 or len(set(visible)) != len(visible):
        raise RuntimeError("use one to three unique nonzero GPUs")
    return visible


def build_trainer(cfg) -> L.Trainer:
    visible = explicit_nonzero_visible_gpus()
    milestones = {int(step): str(name) for step, name in cfg.training.milestones.items()}
    validate_milestones(milestones, int(cfg.training.max_steps))
    output_root = Path(cfg.output.root)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        LatestAdapterCheckpoint(output_root, int(cfg.training.checkpoint_every_n_steps)),
        TierMilestoneCheckpoint(output_root, milestones),
    ]
    return L.Trainer(
        accelerator="gpu",
        devices=len(visible),
        strategy="ddp" if len(visible) > 1 else "auto",
        precision=str(cfg.training.precision),
        max_steps=int(cfg.training.max_steps),
        max_epochs=int(cfg.training.max_epochs),
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        gradient_clip_algorithm="norm",
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    cfg = OmegaConf.load(args.config)
    if args.preflight_batches:
        cfg.output.root = str(Path(cfg.output.root) / "preflight")
        cfg.training.max_steps = 1
        cfg.training.max_epochs = 1
        cfg.training.checkpoint_every_n_steps = 1
        cfg.training.milestones = {1: "recommended"}
        cfg.data.num_workers = 0
    output_root = Path(cfg.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_root / "resolved_train_config.yaml")
    L.seed_everything(int(cfg.training.seed), workers=True)
    dataset = PairedV2VDataset(cfg.data.manifest, sample_frames=int(cfg.data.num_frames))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.data.num_workers) > 0,
        drop_last=True,
    )
    system = TieredPairedV2VSystem(OmegaConf.to_container(cfg, resolve=True))
    trainer = build_trainer(cfg)
    trainer.fit(system, train_dataloaders=loader)


if __name__ == "__main__":
    main()
