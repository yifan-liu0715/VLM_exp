"""Exact RGB mask palettes consumed by the pinned SCAIL-2 WAN code."""

from __future__ import annotations

import numpy as np


BLUE = np.array([0, 0, 255], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)


def _normalize(mask: object) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {value.shape}")
    return np.asarray(value > 0, dtype=bool)


def _encode(mask: object, background: np.ndarray) -> np.ndarray:
    foreground = _normalize(mask)
    output = np.broadcast_to(background, (*foreground.shape, 3)).copy()
    output[foreground] = BLUE
    return output


def encode_reference(mask: object) -> np.ndarray:
    """White background and blue actor, as required for the reference mask."""

    return _encode(mask, WHITE)


def encode_video_frame(mask: object) -> np.ndarray:
    """Black background and blue actor, as required for driving-video masks."""

    return _encode(mask, BLACK)
