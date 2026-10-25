"""Pure media transformations shared by preprocessing and audits."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


ACTOR_RGB = (0, 114, 255)


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if not value.any():
        raise ValueError("mask must contain at least one actor pixel")
    return value


def _validate_actor_rgb(actor_rgb: tuple[int, int, int]) -> np.ndarray:
    color = np.asarray(actor_rgb, dtype=np.int64)
    if color.shape != (3,) or np.any(color < 0) or np.any(color > 255):
        raise ValueError("actor_rgb must contain three uint8-compatible values")
    if np.all(color == 0) or np.all(color == 255):
        raise ValueError("actor color must differ from black and white backgrounds")
    return color.astype(np.uint8)


def _colorize(
    mask: np.ndarray,
    *,
    background_rgb: tuple[int, int, int],
    actor_rgb: tuple[int, int, int],
) -> np.ndarray:
    value = _validate_mask(mask)
    actor = _validate_actor_rgb(actor_rgb)
    output = np.empty((*value.shape, 3), dtype=np.uint8)
    output[...] = np.asarray(background_rgb, dtype=np.uint8)
    output[value] = actor
    return output


def colorize_reference_mask(
    mask: np.ndarray,
    actor_rgb: tuple[int, int, int] = ACTOR_RGB,
) -> np.ndarray:
    return _colorize(mask, background_rgb=(0, 0, 0), actor_rgb=actor_rgb)


def colorize_replacement_mask(
    mask: np.ndarray,
    actor_rgb: tuple[int, int, int] = ACTOR_RGB,
) -> np.ndarray:
    return _colorize(mask, background_rgb=(255, 255, 255), actor_rgb=actor_rgb)


def _resized_shape(
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> tuple[int, int]:
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("source and target dimensions must be positive")
    scale = min(target_height / source_height, target_width / source_width)
    resized_height = min(target_height, max(1, math.floor(source_height * scale + 0.5)))
    resized_width = min(target_width, max(1, math.floor(source_width * scale + 0.5)))
    return resized_height, resized_width


def fit_pad_rgb(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("RGB frame must have shape H,W,3 and dtype uint8")
    resized_height, resized_width = _resized_shape(
        value.shape[0], value.shape[1], int(height), int(width)
    )
    resized = np.asarray(
        Image.fromarray(value).resize(
            (resized_width, resized_height), Image.Resampling.LANCZOS
        )
    )
    remaining_height = int(height) - resized_height
    remaining_width = int(width) - resized_width
    top = remaining_height // 2
    left = remaining_width // 2
    return np.pad(
        resized,
        (
            (top, remaining_height - top),
            (left, remaining_width - left),
            (0, 0),
        ),
        mode="edge",
    )


def fit_pad_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    value = _validate_mask(mask)
    resized_height, resized_width = _resized_shape(
        value.shape[0], value.shape[1], int(height), int(width)
    )
    resized = np.asarray(
        Image.fromarray(value.astype(np.uint8) * 255).resize(
            (resized_width, resized_height), Image.Resampling.NEAREST
        )
    ) > 127
    remaining_height = int(height) - resized_height
    remaining_width = int(width) - resized_width
    top = remaining_height // 2
    left = remaining_width // 2
    return np.pad(
        resized,
        ((top, remaining_height - top), (left, remaining_width - left)),
        mode="constant",
        constant_values=False,
    )


def mask_statistics(mask: np.ndarray) -> dict[str, object]:
    value = _validate_mask(mask)
    ys, xs = np.where(value)
    return {
        "pixels": int(value.sum()),
        "fraction": float(value.mean()),
        "bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
    }
