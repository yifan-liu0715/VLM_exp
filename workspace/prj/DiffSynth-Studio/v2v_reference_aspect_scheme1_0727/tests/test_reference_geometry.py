from __future__ import annotations

import ast
import hashlib
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

from reference_geometry import resize_reference_preserving_geometry
from training_contracts import validate_training_config


class ReferenceGeometryTests(unittest.TestCase):
    def _save(self, array: np.ndarray, name: str) -> Path:
        path = Path(self.temporary.name) / name
        Image.fromarray(array, mode="RGB").save(path)
        return path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_returns_exact_model_shape_for_all_current_buckets(self) -> None:
        image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        image[180:844, 390:634] = (20, 80, 200)
        path = self._save(image, "portrait.png")

        for height, width in ((256, 704), (384, 704), (704, 384)):
            with self.subTest(height=height, width=width):
                result = resize_reference_preserving_geometry(
                    path,
                    height=height,
                    width=width,
                )
                self.assertEqual(result.shape, (3, height, width))
                self.assertEqual(result.dtype, torch.float32)
                self.assertGreaterEqual(float(result.min()), -1.0)
                self.assertLessEqual(float(result.max()), 1.0)

    def test_foreground_aspect_ratio_is_not_stretched(self) -> None:
        image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        image[160:864, 392:632] = (10, 40, 180)
        path = self._save(image, "tall_subject.png")

        result = resize_reference_preserving_geometry(
            path,
            height=256,
            width=704,
        )
        rgb = (
            result.add(1)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        subject = (rgb[..., 2] > 120) & (rgb[..., 0] < 80)
        ys, xs = np.where(subject)
        observed_ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        original_ratio = 240 / 704
        self.assertAlmostEqual(observed_ratio, original_ratio, delta=0.025)

    def test_all_white_reference_is_deterministic(self) -> None:
        image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        path = self._save(image, "white.png")

        first = resize_reference_preserving_geometry(path, 384, 704)
        second = resize_reference_preserving_geometry(path, 384, 704)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first == 1.0))

    def test_source_target_preprocessing_contract_is_unchanged(self) -> None:
        dataset_path = SCRIPTS / "teacher_aligned_dataset.py"
        source = dataset_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        expected = {
            "target_size": "ffdd4c00302605c8e2adbbabf0782d07344a8a1cb4453b7d1f7d6f084a2f8b53",
            "_resize_video": "a3513e16a5a3683c434cc2008f317b19be8d6c1c3b925b3d26d561fe455f7750",
        }
        observed: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in expected:
                segment = ast.get_source_segment(source, node)
                observed[node.name] = hashlib.sha256(segment.encode()).hexdigest()
        self.assertEqual(observed, expected)

    def test_scheme1_contract_requires_exactly_2000_steps(self) -> None:
        config = {
            "data": {
                "train_manifest": "/experiment/metadata/train_manifest.csv",
                "num_frames": 97,
                "max_long_side": 704,
            },
            "lora": {"rank": 96, "alpha": 96},
            "training": {
                "run_kind": "reference_aspect_scheme1",
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
