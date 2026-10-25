"""Aspect-preserving geometry shared by Scheme 4 training and inference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional


SCHEME4_CANVASES = {
    "1280x512": (384, 960),
    "1280x704": (352, 640),
    "704x1280": (640, 352),
}


def _validate_canvas(height: int, width: int) -> tuple[int, int]:
    height, width = int(height), int(width)
    if min(height, width) <= 0:
        raise ValueError("canvas dimensions must be positive")
    if height % 32 or width % 32:
        raise ValueError(
            f"Scheme 4 canvas must be 32-aligned: {width}x{height}"
        )
    return height, width


def scheme4_canvas(bucket: str) -> tuple[int, int]:
    try:
        canvas = SCHEME4_CANVASES[str(bucket)]
    except KeyError as error:
        raise ValueError(f"unsupported Scheme 4 bucket: {bucket}") from error
    return _validate_canvas(*canvas)


def _require_same_aspect(
    source_height: int,
    source_width: int,
    height: int,
    width: int,
) -> None:
    if int(source_width) * int(height) != int(source_height) * int(width):
        raise ValueError(
            "video resize must preserve the exact native aspect ratio: "
            f"{source_width}x{source_height} -> {width}x{height}"
        )


def resize_video_to_scheme4_canvas(
    frames: np.ndarray,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize T,H,W,C video uniformly to C,T,H,W without cropping."""
    height, width = _validate_canvas(height, width)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected T,H,W,3 frames, got {tuple(frames.shape)}")
    source_height, source_width = map(int, frames.shape[1:3])
    _require_same_aspect(
        source_height,
        source_width,
        height,
        width,
    )
    tensor = (
        torch.from_numpy(frames)
        .to(torch.float32)
        .permute(0, 3, 1, 2)
        .div(127.5)
        .sub(1.0)
    )
    resized = functional.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.permute(1, 0, 2, 3).contiguous().clamp(-1.0, 1.0)


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def resize_reference_to_scheme4_canvas(
    path: str | Path,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Aspect-fit an RGB Reference and replicate-pad to the video canvas."""
    height, width = _validate_canvas(height, width)
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
    source_height, source_width = map(int, tensor.shape[-2:])
    scale = min(height / source_height, width / source_width)
    resized_height = min(
        height,
        max(1, _round_half_up(source_height * scale)),
    )
    resized_width = min(
        width,
        max(1, _round_half_up(source_width * scale)),
    )
    resized = functional.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    remaining_height = height - resized_height
    remaining_width = width - resized_width
    top = remaining_height // 2
    left = remaining_width // 2
    output = functional.pad(
        resized,
        (
            left,
            remaining_width - left,
            top,
            remaining_height - top,
        ),
        mode="replicate",
    )
    if output.shape[-2:] != (height, width):
        raise RuntimeError(
            f"Reference output is {tuple(output.shape[-2:])}, "
            f"expected {(height, width)}"
        )
    return output.squeeze(0).contiguous().clamp(-1.0, 1.0)
