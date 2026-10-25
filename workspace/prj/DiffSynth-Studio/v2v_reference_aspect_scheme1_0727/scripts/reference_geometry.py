"""Aspect-preserving preprocessing for external V2V reference images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch


WHITE_THRESHOLD = 245
WHITE_BORDER_FRACTION = 0.70
FOREGROUND_MARGIN = 0.12


def _border_white_fraction(array: np.ndarray) -> float:
    height, width = array.shape[:2]
    thickness = max(1, round(min(height, width) * 0.02))
    border = np.concatenate(
        (
            array[:thickness].reshape(-1, 3),
            array[-thickness:].reshape(-1, 3),
            array[:, :thickness].reshape(-1, 3),
            array[:, -thickness:].reshape(-1, 3),
        ),
        axis=0,
    )
    return float(np.mean(np.all(border >= WHITE_THRESHOLD, axis=1)))


def _foreground_box(array: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = np.any(array < WHITE_THRESHOLD, axis=2)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    margin_x = round((right - left) * FOREGROUND_MARGIN)
    margin_y = round((bottom - top) * FOREGROUND_MARGIN)
    return (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(image_width, right + margin_x),
        min(image_height, bottom + margin_y),
    )


def _largest_aspect_crop(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    target_ratio = target_width / target_height
    if image_width / image_height >= target_ratio:
        crop_height = image_height
        crop_width = min(image_width, round(crop_height * target_ratio))
    else:
        crop_width = image_width
        crop_height = min(image_height, round(crop_width / target_ratio))
    return max(1, crop_width), max(1, crop_height)


def _centered_crop_box(
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
    center_x: float,
    center_y: float,
) -> tuple[int, int, int, int]:
    left = round(center_x - crop_width / 2)
    top = round(center_y - crop_height / 2)
    left = min(max(0, left), image_width - crop_width)
    top = min(max(0, top), image_height - crop_height)
    return left, top, left + crop_width, top + crop_height


def _fit_on_white_canvas(
    image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    fitted = ImageOps.contain(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = ((width - fitted.width) // 2, (height - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def resize_reference_preserving_geometry(
    path: str | Path,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return a `C,H,W` tensor without non-uniformly stretching the subject."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    array = np.asarray(image)
    image_width, image_height = image.size
    crop_width, crop_height = _largest_aspect_crop(
        image_width,
        image_height,
        width,
        height,
    )

    foreground = _foreground_box(array)
    has_white_border = _border_white_fraction(array) >= WHITE_BORDER_FRACTION
    if foreground is not None and has_white_border:
        foreground = _expand_box(foreground, image_width, image_height)
        foreground_width = foreground[2] - foreground[0]
        foreground_height = foreground[3] - foreground[1]
        if foreground_width <= crop_width and foreground_height <= crop_height:
            center_x = (foreground[0] + foreground[2]) / 2
            center_y = (foreground[1] + foreground[3]) / 2
            crop_box = _centered_crop_box(
                image_width,
                image_height,
                crop_width,
                crop_height,
                center_x,
                center_y,
            )
            output = image.crop(crop_box).resize(
                (width, height),
                resample=Image.Resampling.LANCZOS,
            )
        else:
            output = _fit_on_white_canvas(
                image.crop(foreground),
                width,
                height,
            )
    else:
        crop_box = _centered_crop_box(
            image_width,
            image_height,
            crop_width,
            crop_height,
            image_width / 2,
            image_height / 2,
        )
        output = image.crop(crop_box).resize(
            (width, height),
            resample=Image.Resampling.LANCZOS,
        )

    tensor = torch.from_numpy(np.asarray(output).copy()).to(torch.float32)
    return tensor.permute(2, 0, 1).div(127.5).sub(1.0).contiguous()
