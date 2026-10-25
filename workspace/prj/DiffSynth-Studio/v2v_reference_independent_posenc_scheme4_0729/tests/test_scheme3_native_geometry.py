from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from native_geometry import (
    native_canvas,
    normalize_video_native,
    resize_reference_to_native_canvas,
)


class Scheme3NativeGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _save_reference(self, array: np.ndarray) -> Path:
        path = Path(self.temporary.name) / "reference.png"
        Image.fromarray(array, mode="RGB").save(path)
        return path

    def test_native_buckets_are_not_resized(self) -> None:
        self.assertEqual(native_canvas(512, 1280), (512, 1280))
        self.assertEqual(native_canvas(704, 1280), (704, 1280))
        self.assertEqual(native_canvas(1280, 704), (1280, 704))

    def test_misaligned_native_canvas_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "32-aligned"):
            native_canvas(513, 1280)

    def test_video_normalization_preserves_native_shape(self) -> None:
        frames = np.zeros((97, 512, 1280, 3), dtype=np.uint8)
        tensor = normalize_video_native(frames)

        self.assertEqual(tuple(tensor.shape), (3, 97, 512, 1280))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(float(tensor.min()), -1.0)
        self.assertEqual(float(tensor.max()), -1.0)

    def test_reference_matches_video_canvas_without_stretching(self) -> None:
        image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        image[160:864, 392:632] = (10, 40, 180)
        path = self._save_reference(image)

        output = resize_reference_to_native_canvas(
            path,
            height=512,
            width=1280,
        )

        self.assertEqual(tuple(output.shape), (3, 512, 1280))
        rgb = (
            output.add(1)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        subject = (rgb[..., 2] > 120) & (rgb[..., 0] < 80)
        ys, xs = np.where(subject)
        observed_ratio = (xs.max() - xs.min() + 1) / (
            ys.max() - ys.min() + 1
        )
        original_ratio = 240 / 704
        self.assertAlmostEqual(observed_ratio, original_ratio, delta=0.02)


if __name__ == "__main__":
    unittest.main()
