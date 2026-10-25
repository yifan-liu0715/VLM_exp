"""Shared aspect-fit and minimum-padding geometry for Scheme 2 inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class FitPlan:
    """Deterministic resize and symmetric padding parameters."""

    resized_h: int
    resized_w: int
    top: int
    bottom: int
    left: int
    right: int


def _positive_dimensions(*values: int) -> None:
    if any(int(value) <= 0 for value in values):
        raise ValueError("all dimensions and alignment values must be positive")


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def nearest_aligned_canvas(
    native_h: int,
    native_w: int,
    max_long_side: int,
    align: int = 32,
) -> tuple[int, int]:
    """Return the nearest aligned canvas after limiting the native long side."""
    _positive_dimensions(native_h, native_w, max_long_side, align)
    scale = min(1.0, float(max_long_side) / max(native_h, native_w))
    scaled_h = float(native_h) * scale
    scaled_w = float(native_w) * scale

    def aligned(value: float) -> int:
        units = _round_half_up(value / align)
        return max(int(align), units * int(align))

    return aligned(scaled_h), aligned(scaled_w)


def aspect_fit_plan(
    native_h: int,
    native_w: int,
    canvas_h: int,
    canvas_w: int,
) -> FitPlan:
    """Calculate a uniform fit and the smallest symmetric padding."""
    _positive_dimensions(native_h, native_w, canvas_h, canvas_w)
    scale = min(
        float(canvas_h) / float(native_h),
        float(canvas_w) / float(native_w),
    )
    resized_h = min(canvas_h, max(1, _round_half_up(native_h * scale)))
    resized_w = min(canvas_w, max(1, _round_half_up(native_w * scale)))
    remaining_h = canvas_h - resized_h
    remaining_w = canvas_w - resized_w
    top = remaining_h // 2
    left = remaining_w // 2
    return FitPlan(
        resized_h=resized_h,
        resized_w=resized_w,
        top=top,
        bottom=remaining_h - top,
        left=left,
        right=remaining_w - left,
    )


def _resize_and_pad_nchw(
    tensor: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"expected N,C,H,W tensor, got {tuple(tensor.shape)}")
    plan = aspect_fit_plan(
        int(tensor.shape[-2]),
        int(tensor.shape[-1]),
        int(height),
        int(width),
    )
    resized = functional.interpolate(
        tensor,
        size=(plan.resized_h, plan.resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    output = functional.pad(
        resized,
        (plan.left, plan.right, plan.top, plan.bottom),
        mode="replicate",
    )
    if output.shape[-2:] != (height, width):
        raise RuntimeError(
            f"geometry output is {tuple(output.shape[-2:])}, "
            f"expected {(height, width)}"
        )
    return output.clamp(-1.0, 1.0).contiguous()


def resize_video_preserving_geometry(
    frames: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert T,H,W,C uint8 frames to normalized C,T,H,W."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected T,H,W,3 frames, got {tuple(frames.shape)}")
    tensor = torch.from_numpy(frames).to(torch.float32)
    tensor = tensor.permute(0, 3, 1, 2).div(127.5).sub(1.0)
    output = _resize_and_pad_nchw(tensor, height, width)
    return output.permute(1, 0, 2, 3).contiguous()


def resize_reference_preserving_geometry(
    path: str | Path,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert one RGB image to normalized C,H,W using the shared geometry."""
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB")).copy()
    tensor = torch.from_numpy(image).to(torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1.0)
    output = _resize_and_pad_nchw(tensor, height, width)
    return output.squeeze(0).contiguous()
