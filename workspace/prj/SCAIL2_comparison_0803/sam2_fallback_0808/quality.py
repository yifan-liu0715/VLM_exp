"""Deterministic mask statistics used by the fail-closed quality gate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def clean_reference_components(mask: object, minimum_ratio: float = 0.001) -> np.ndarray:
    """Remove tiny connected components and require one remaining actor."""

    import cv2

    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError(f"reference mask must be 2-D, got {value.shape}")
    binary = np.asarray(value > 0, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum_pixels = max(1, int(np.ceil(binary.size * minimum_ratio)))
    keep = [index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_pixels]
    if len(keep) != 1:
        raise ValueError(f"expected one reference component, got {len(keep)}")
    return np.asarray(labels == keep[0], dtype=bool)


def validate_human_review(payload: dict) -> None:
    required = {"status", "frames", "id_switch", "large_background_absorption"}
    if set(payload) != required:
        raise ValueError(f"qc_review keys must be exactly {sorted(required)}")
    if payload["status"] != "passed":
        raise ValueError("qc_review status must be passed")
    if payload["frames"] != [0, 48, 96]:
        raise ValueError("qc_review frames must be [0, 48, 96]")
    if payload["id_switch"] is not False:
        raise ValueError("qc_review id_switch must be false")
    if payload["large_background_absorption"] is not False:
        raise ValueError("qc_review large_background_absorption must be false")


def compute_mask_stats(masks: Sequence[object], jump_threshold: float = 0.50) -> dict:
    if not masks:
        raise ValueError("no masks supplied")
    if not 0 < jump_threshold <= 1:
        raise ValueError("jump_threshold must be in (0, 1]")

    normalized: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for index, value in enumerate(masks):
        mask = np.asarray(value)
        if mask.ndim != 2:
            raise ValueError(f"mask at frame {index} must be 2-D, got {mask.shape}")
        mask = np.asarray(mask > 0, dtype=bool)
        if expected_shape is None:
            expected_shape = mask.shape
        elif mask.shape != expected_shape:
            raise ValueError(
                f"mask shape mismatch at frame {index}: {mask.shape} != {expected_shape}"
            )
        normalized.append(mask)

    assert expected_shape is not None
    canvas_pixels = int(np.prod(expected_shape))
    areas = [int(mask.sum(dtype=np.int64)) for mask in normalized]
    ratios = [float(area / canvas_pixels) for area in areas]
    empty_frames = [index for index, area in enumerate(areas) if area == 0]
    jump_frames: list[int] = []
    relative_area_changes = [0.0]
    for index in range(1, len(areas)):
        change = float(abs(areas[index] - areas[index - 1]) / max(areas[index - 1], 1))
        relative_area_changes.append(change)
        if change > jump_threshold:
            jump_frames.append(index)

    return {
        "num_frames": len(normalized),
        "shape": list(expected_shape),
        "pixel_areas": areas,
        "area_ratios": ratios,
        "relative_area_changes": relative_area_changes,
        "empty_frames": empty_frames,
        "jump_frames": jump_frames,
        "jump_threshold": float(jump_threshold),
        "automatic_pass": not empty_frames and not jump_frames,
    }
