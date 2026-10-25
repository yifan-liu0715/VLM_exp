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
    run_kind = str(training.get("run_kind", "formal_teacher_aligned"))
    expected_steps = {
        "formal_teacher_aligned": 20_000,
        "reference_aspect_scheme2": 2_000,
        "native_resolution_scheme3": 2_000,
        "reference_independent_posenc_scheme4_gate": 1_000,
    }
    if run_kind not in expected_steps:
        raise ValueError(f"unsupported training run_kind={run_kind}")
    if int(training["max_steps"]) != expected_steps[run_kind]:
        raise ValueError(
            f"{run_kind} run must use max_steps={expected_steps[run_kind]}"
        )
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
    if run_kind == "reference_independent_posenc_scheme4_gate":
        if int(data.get("bucket_align", 0)) != 32:
            raise ValueError(
                "reference_independent_posenc_scheme4_gate requires "
                "bucket_align=32"
            )
        expected_canvases = {
            "1280x512": [384, 960],
            "1280x704": [352, 640],
            "704x1280": [640, 352],
        }
        if data.get("canvas_by_bucket") != expected_canvases:
            raise ValueError(
                "Scheme 4 must use the documented aspect-preserving canvases"
            )
        reference_position = config["reference_position"]
        if (
            reference_position.get("enabled") is not True
            or
            int(reference_position["max_frames"]) != 4
            or int(reference_position["max_height"]) != 64
            or int(reference_position["max_width"]) != 64
            or float(reference_position["init_std"]) != 0.02
        ):
            raise ValueError(
                "Scheme 4 Reference position encoding contract mismatch"
            )
    elif run_kind == "native_resolution_scheme3":
        if data.get("native_resolution") is not True:
            raise ValueError(
                "native_resolution_scheme3 requires native_resolution: true"
            )
        if "max_long_side" in data:
            raise ValueError(
                "native_resolution_scheme3 forbids max_long_side"
            )
        if int(data.get("bucket_align", 0)) != 32:
            raise ValueError(
                "native_resolution_scheme3 requires bucket_align=32"
            )
    elif int(data["max_long_side"]) not in (1088, 704, 640):
        raise ValueError("max_long_side must be one of the preflight values")


class PathLike:
    """Small path-name helper that avoids filesystem access in config validation."""

    def __init__(self, value: object):
        self.value = str(value).replace("\\", "/").rstrip("/")

    @property
    def name(self) -> str:
        return self.value.rsplit("/", 1)[-1]
