#!/usr/bin/env python3
"""Render original, legacy-stretched, and Scheme-1 Reference previews."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as functional

from reference_geometry import resize_reference_preserving_geometry


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "preflight" / "reference_preprocess_comparison.png"
SAMPLES = (
    (
        "1280x512 / mymotion__001",
        Path(
            "/nvmedata/workspace2/users/nina/dataset_video_train/"
            "1280x512/reference/mymotion__001.jpg"
        ),
        256,
        704,
    ),
    (
        "1280x704 / mymotion__002",
        Path(
            "/nvmedata/workspace2/users/nina/dataset_video_train/"
            "1280x704/reference/mymotion__002.jpg"
        ),
        384,
        704,
    ),
    (
        "704x1280 / mymotion__humandance2__batch1_mote2_1__1007",
        Path(
            "/nvmedata/workspace2/users/nina/dataset_video_train/"
            "704x1280/reference/"
            "mymotion__humandance2__batch1_mote2_1__1007.jpg"
        ),
        704,
        384,
    ),
)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.add(1)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def legacy_resize(path: Path, height: int, width: int) -> Image.Image:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB")).copy()
    tensor = torch.from_numpy(array).to(torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    tensor = functional.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return tensor_to_image(tensor.squeeze(0))


def labeled(image: Image.Image, label: str, cell_size=(704, 704)) -> Image.Image:
    canvas = Image.new("RGB", cell_size, color="white")
    fitted = image.copy()
    fitted.thumbnail((cell_size[0], cell_size[1] - 42), Image.Resampling.LANCZOS)
    canvas.paste(
        fitted,
        ((cell_size[0] - fitted.width) // 2, 42 + (cell_size[1] - 42 - fitted.height) // 2),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, cell_size[0], 42), fill="black")
    draw.text((12, 13), label, fill="white")
    return canvas


def main() -> None:
    rows = []
    for name, path, height, width in SAMPLES:
        with Image.open(path) as handle:
            original = handle.convert("RGB")
        old = legacy_resize(path, height, width)
        new = tensor_to_image(
            resize_reference_preserving_geometry(path, height, width)
        )
        rows.append(
            (
                labeled(original, f"{name} | ORIGINAL 1024x1024"),
                labeled(old, f"LEGACY STRETCH -> {width}x{height}"),
                labeled(new, f"SCHEME 1 -> {width}x{height}"),
            )
        )
    output = Image.new("RGB", (704 * 3, 704 * len(rows)), color="white")
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            output.paste(cell, (704 * column_index, 704 * row_index))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
