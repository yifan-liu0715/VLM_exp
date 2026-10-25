"""Native-resolution geometry shared by Scheme 3 training and inference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional


def native_canvas(
    native_h: int,
    native_w: int,
    align: int = 32,
) -> tuple[int, int]:
    """Validate and return an unchanged native video canvas."""
    if min(int(native_h), int(native_w), int(align)) <= 0:
        raise ValueError("native dimensions and alignment must be positive")
    if int(native_h) % int(align) or int(native_w) % int(align):
        raise ValueError(
            f"native video dimensions must be {int(align)}-aligned: "
            f"{int(native_w)}x{int(native_h)}"
        )
    return int(native_h), int(native_w)


def normalize_video_native(frames: np.ndarray) -> torch.Tensor:
    """Convert T,H,W,C uint8 frames to normalized C,T,H,W without resizing."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected T,H,W,3 frames, got {tuple(frames.shape)}")
    tensor = torch.from_numpy(frames).to(torch.float32)
    output = tensor.permute(0, 3, 1, 2).div(127.5).sub(1.0)
    return output.permute(1, 0, 2, 3).contiguous()


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def resize_reference_to_native_canvas(
    path: str | Path,
    height: int,
    width: int,
) -> torch.Tensor:
    """Aspect-fit an RGB reference and replicate-pad it to the native canvas."""
    native_canvas(int(height), int(width))
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB")).copy()
    tensor = (
        torch.from_numpy(image)
        .to(torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .div(127.5)
        .sub(1.0)
    )
    source_h, source_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    scale = min(float(height) / source_h, float(width) / source_w)
    resized_h = min(int(height), max(1, _round_half_up(source_h * scale)))
    resized_w = min(int(width), max(1, _round_half_up(source_w * scale)))
    resized = functional.interpolate(
        tensor,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    remaining_h = int(height) - resized_h
    remaining_w = int(width) - resized_w
    top = remaining_h // 2
    left = remaining_w // 2
    output = functional.pad(
        resized,
        (
            left,
            remaining_w - left,
            top,
            remaining_h - top,
        ),
        mode="replicate",
    )
    if output.shape[-2:] != (int(height), int(width)):
        raise RuntimeError(
            f"reference output is {tuple(output.shape[-2:])}, "
            f"expected {(int(height), int(width))}"
        )
    return output.squeeze(0).clamp(-1.0, 1.0).contiguous()
