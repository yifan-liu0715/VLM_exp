"""Small, testable adapters around the official SAM2 predictors.

This module deliberately contains no dataset discovery. Callers must obtain inputs
through :mod:`sam2_fallback_0808.inputs`, which enforces the target-leakage boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_bool_mask(mask: object, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D mask, got {array.shape}")
    return np.asarray(array > 0, dtype=bool)


def segment_reference(predictor, image_rgb: np.ndarray, box_xyxy) -> np.ndarray:
    """Segment one reference actor from a verified in-canvas XYXY box."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"reference image must be HxWx3 RGB, got {image.shape}")
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(1, 4)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box[0]
    if not np.isfinite(box).all() or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"reference box outside canvas: {tuple(map(float, box[0]))}")

    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    score_array = np.asarray(scores).reshape(-1)
    if score_array.size == 0 or not np.isfinite(score_array).all():
        raise ValueError("reference predictor returned invalid scores")
    mask_array = np.asarray(masks)
    if mask_array.ndim < 3 or mask_array.shape[0] != score_array.size:
        raise ValueError("reference predictor returned incompatible masks and scores")
    mask = _as_bool_mask(mask_array[int(np.argmax(score_array))], name="reference mask")
    if mask.shape != (height, width) or not mask.any():
        raise ValueError(f"invalid reference mask: shape={mask.shape}, nonempty={mask.any()}")
    return mask


def mask_iou(left: object, right: object) -> float:
    a = _as_bool_mask(left, name="left mask")
    b = _as_bool_mask(right, name="right mask")
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} != {b.shape}")
    union = np.logical_or(a, b).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum(dtype=np.int64)
    return float(intersection / union)


def select_source_track(
    reference_mask: object, candidate_masks: Sequence[object]
) -> tuple[int, list[float]]:
    """Choose the first-frame source actor with greatest reference-mask IoU."""

    if not candidate_masks:
        raise ValueError("no source track candidates")
    scores = [mask_iou(reference_mask, candidate) for candidate in candidate_masks]
    return int(np.argmax(np.asarray(scores, dtype=np.float64))), scores


def validate_propagated_masks(
    masks: Sequence[object], expected_frames: int, expected_shape: tuple[int, int]
) -> list[np.ndarray]:
    """Validate every propagated frame and return normalized boolean masks."""

    if len(masks) != expected_frames:
        raise ValueError(f"expected {expected_frames} masks, got {len(masks)}")
    normalized: list[np.ndarray] = []
    for index, value in enumerate(masks):
        mask = _as_bool_mask(value, name=f"mask at frame {index}")
        if mask.shape != expected_shape:
            raise ValueError(f"shape mismatch at frame {index}: {mask.shape}")
        if not mask.any():
            raise ValueError(f"empty mask at frame {index}")
        normalized.append(mask)
    return normalized
