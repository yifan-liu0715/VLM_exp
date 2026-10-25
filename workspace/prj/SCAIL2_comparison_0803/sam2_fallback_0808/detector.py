from __future__ import annotations

"""YOLOX person detection adapted from the official DWPose ONNX branch.

The preprocessing and decoding equations follow IDEA-Research/DWPose commit
3dca5db79d9f9ffdd378753ddf6ec66535aace88 (Apache-2.0).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int


def select_single_or_manual(
    detections: Sequence[Detection],
    canvas: tuple[int, int],
    manual_box: tuple[float, float, float, float] | None = None,
) -> Detection:
    width, height = canvas
    if manual_box is not None:
        x1, y1, x2, y2 = map(float, manual_box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("manual box lies outside the source canvas")
        return Detection((x1, y1, x2, y2), 1.0, -1)

    people = [item for item in detections if item.class_id == 0]
    if not people:
        raise ValueError("no person detection")
    if len(people) > 1:
        raise ValueError("multiple people require first-frame mask IoU matching")
    return people[0]


def nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1.0)
        height = np.maximum(0.0, yy2 - yy1 + 1.0)
        overlap = width * height
        iou = overlap / (areas[index] + areas[order[1:]] - overlap)
        order = order[np.where(iou <= nms_thr)[0] + 1]
    return keep


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    nms_thr: float,
    score_thr: float,
) -> np.ndarray:
    final: list[np.ndarray] = []
    for class_id in range(scores.shape[1]):
        class_scores = scores[:, class_id]
        valid = class_scores > score_thr
        if not valid.any():
            continue
        valid_boxes = boxes[valid]
        valid_scores = class_scores[valid]
        keep = nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            labels = np.full((len(keep), 1), class_id, dtype=np.float32)
            final.append(
                np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], labels], axis=1
                )
            )
    if not final:
        return np.empty((0, 6), dtype=np.float32)
    return np.concatenate(final, axis=0)


def _preprocess(image_bgr: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    input_height, input_width = input_size
    height, width = image_bgr.shape[:2]
    ratio = min(input_height / height, input_width / width)
    resized = cv2.resize(
        image_bgr,
        (int(width * ratio), int(height * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    tensor = padded.transpose(2, 0, 1).astype(np.float32, copy=False)
    return np.ascontiguousarray(tensor), ratio


def _decode(outputs: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []
    for stride in (8, 16, 32):
        height, width = input_size[0] // stride, input_size[1] // stride
        yv, xv = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grid = np.concatenate(grids, axis=1)
    strides = np.concatenate(expanded_strides, axis=1)
    decoded = outputs.copy()
    decoded[..., :2] = (decoded[..., :2] + grid) * strides
    decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * strides
    return decoded


def detect_people(
    image_rgb: np.ndarray,
    model_path: Path,
    score_threshold: float = 0.30,
) -> list[Detection]:
    import onnxruntime as ort

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected RGB HxWx3 image, got {image_rgb.shape}")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    input_height = int(shape[2]) if isinstance(shape[2], int) else 640
    input_width = int(shape[3]) if isinstance(shape[3], int) else 640
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    tensor, ratio = _preprocess(image_bgr, (input_height, input_width))
    raw = session.run(None, {model_input.name: tensor[None, ...]})[0]
    predictions = _decode(np.asarray(raw), (input_height, input_width))[0]

    boxes = predictions[:, :4]
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy /= ratio
    scores = predictions[:, 4:5] * predictions[:, 5:]
    detections = multiclass_nms(boxes_xyxy, scores, 0.45, score_threshold)

    height, width = image_rgb.shape[:2]
    result: list[Detection] = []
    for x1, y1, x2, y2, score, class_id in detections:
        if int(class_id) != 0 or not np.isfinite([x1, y1, x2, y2, score]).all():
            continue
        clipped = (
            float(np.clip(x1, 0, width)),
            float(np.clip(y1, 0, height)),
            float(np.clip(x2, 0, width)),
            float(np.clip(y2, 0, height)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        result.append(Detection(clipped, float(score), 0))
    return sorted(result, key=lambda item: item.score, reverse=True)
