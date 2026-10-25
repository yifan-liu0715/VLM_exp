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

from aspect_fit_geometry import (
    aspect_fit_plan,
    nearest_aligned_canvas,
    resize_reference_preserving_geometry,
    resize_video_preserving_geometry,
)
from training_contracts import validate_training_config


class Scheme2GeometryTests(unittest.TestCase):
    def _save(self, array: np.ndarray, name: str) -> Path:
        path = Path(self.temporary.name) / name
        Image.fromarray(array, mode="RGB").save(path)
        return path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_nearest_aligned_canvas_for_all_native_buckets(self) -> None:
        self.assertEqual(
            nearest_aligned_canvas(512, 1280, 704, 32),
            (288, 704),
        )
        self.assertEqual(
            nearest_aligned_canvas(704, 1280, 704, 32),
            (384, 704),
        )
        self.assertEqual(
            nearest_aligned_canvas(1280, 704, 704, 32),
            (704, 384),
        )

    def test_aspect_fit_plan_uses_symmetric_minimum_padding(self) -> None:
        wide = aspect_fit_plan(512, 1280, 288, 704)
        self.assertEqual(
            (wide.resized_h, wide.resized_w),
            (282, 704),
        )
        self.assertEqual(
            (wide.top, wide.bottom, wide.left, wide.right),
            (3, 3, 0, 0),
        )

        portrait = aspect_fit_plan(1280, 704, 704, 384)
        self.assertEqual(
            (portrait.resized_h, portrait.resized_w),
            (698, 384),
        )
        self.assertEqual(
            (portrait.top, portrait.bottom, portrait.left, portrait.right),
            (3, 3, 0, 0),
        )

    def test_video_preserves_subject_ratio_and_replicates_padding(self) -> None:
        frame = np.full((512, 1280, 3), 40, dtype=np.uint8)
        frame[96:416, 480:800] = (10, 40, 220)
        frames = np.stack((frame, frame, frame))

        result = resize_video_preserving_geometry(frames, 288, 704)

        self.assertEqual(result.shape, (3, 3, 288, 704))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result[:, :, 0], result[:, :, 3]))
        self.assertTrue(torch.equal(result[:, :, -1], result[:, :, -4]))
        rgb = (
            result[:, 0]
            .add(1)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        subject = (rgb[..., 2] > 150) & (rgb[..., 0] < 80)
        ys, xs = np.where(subject)
        observed_ratio = (xs.max() - xs.min() + 1) / (
            ys.max() - ys.min() + 1
        )
        self.assertAlmostEqual(observed_ratio, 1.0, delta=0.02)

    def test_reference_uses_same_canvas_and_is_deterministic(self) -> None:
        image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        image[160:864, 392:632] = (10, 40, 180)
        path = self._save(image, "portrait.png")

        first = resize_reference_preserving_geometry(path, 288, 704)
        second = resize_reference_preserving_geometry(path, 288, 704)

        self.assertEqual(first.shape, (3, 288, 704))
        self.assertTrue(torch.equal(first, second))
        rgb = (
            first.add(1)
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

    def test_antialiased_resize_stays_in_normalized_range(self) -> None:
        frame = np.zeros((704, 1280, 3), dtype=np.uint8)
        frame[0, :, 0] = 2
        frames = np.stack((frame, frame))

        result = resize_video_preserving_geometry(frames, 384, 704)

        self.assertGreaterEqual(float(result.min()), -1.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            nearest_aligned_canvas(0, 1280, 704, 32)
        with self.assertRaisesRegex(ValueError, "positive"):
            aspect_fit_plan(512, 1280, 0, 704)

    def test_scheme2_contract_requires_exactly_2000_steps(self) -> None:
        config = {
            "data": {
                "train_manifest": "/experiment/metadata/train_manifest.csv",
                "num_frames": 97,
                "max_long_side": 704,
            },
            "lora": {"rank": 96, "alpha": 96},
            "training": {
                "run_kind": "reference_aspect_scheme2",
                "max_steps": 2000,
                "text_drop_prob": 0.1,
                "reference_drop_prob": 0.5,
            },
            "output": {"root": "/experiment"},
        }
        validate_training_config(config)
        config["training"]["max_steps"] = 1999
        with self.assertRaisesRegex(ValueError, "2000"):
            validate_training_config(config)


if __name__ == "__main__":
    unittest.main()
