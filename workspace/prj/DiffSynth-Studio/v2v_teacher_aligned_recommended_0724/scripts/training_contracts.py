"""Pure contracts shared by teacher-aligned training and launch scripts."""

from __future__ import annotations

import math
from collections.abc import Mapping


def explicit_nonzero_visible_gpus(value: str) -> list[str]:
    visible = [item.strip() for item in value.split(",") if item.strip()]
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must list nonzero physical GPUs")
    if any(not item.isdigit() for item in visible):
        raise RuntimeError("GPU identifiers must be non-negative integers")
    if "0" in visible:
        raise RuntimeError("physical GPU0 is forbidden")
    if len(visible) > 3:
        raise RuntimeError("at most three physical GPUs may be used")
    if len(set(visible)) != len(visible):
        raise RuntimeError("duplicate physical GPU identifiers are forbidden")
    return visible


def scaled_learning_rate(
    base_lr: float,
    accumulate_grad_batches: int,
    num_gpus: int,
    num_nodes: int = 1,
) -> float:
    if min(base_lr, accumulate_grad_batches, num_gpus, num_nodes) <= 0:
        raise ValueError("learning-rate scaling inputs must be positive")
    return float(base_lr) * math.sqrt(
        int(accumulate_grad_batches) * int(num_gpus) * int(num_nodes)
    )


def teacher_lr_multiplier(step: int, warmup_steps: int) -> float:
    if step < 0 or warmup_steps < 0:
        raise ValueError("step and warmup_steps must be non-negative")
    if warmup_steps == 0:
        return 1.0
    return min(float(step) / float(warmup_steps), 1.0)


def validate_training_config(config: Mapping) -> None:
    data = config["data"]
    lora = config["lora"]
    training = config["training"]
    if int(training["max_steps"]) != 20_000:
        raise ValueError("formal teacher-aligned run must use max_steps=20000")
    if int(data["num_frames"]) != 97:
        raise ValueError("formal teacher-aligned run must use 97 frames")
    if PathLike(data["train_manifest"]).name != "train_manifest.csv":
        raise ValueError("training must consume metadata/train_manifest.csv")
    if int(lora["rank"]) != 96 or int(lora["alpha"]) != 96:
        raise ValueError("teacher-aligned LoRA must use rank=alpha=96")
    if float(training["text_drop_prob"]) != 0.1:
        raise ValueError("teacher-aligned text_drop_prob must be 0.1")
    if float(training["reference_drop_prob"]) != 0.5:
        raise ValueError("teacher-aligned reference_drop_prob must be 0.5")
    if int(data["max_long_side"]) not in (1088, 704, 640):
        raise ValueError("max_long_side must be one of the preflight values")


class PathLike:
    """Small path-name helper that avoids filesystem access in config validation."""

    def __init__(self, value: object):
        self.value = str(value).replace("\\", "/").rstrip("/")

    @property
    def name(self) -> str:
        return self.value.rsplit("/", 1)[-1]
