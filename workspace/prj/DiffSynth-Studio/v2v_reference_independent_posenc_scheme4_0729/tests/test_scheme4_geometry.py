from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scheme4_geometry import (  # noqa: E402
    resize_reference_to_scheme4_canvas,
    resize_video_to_scheme4_canvas,
    scheme4_canvas,
)
from teacher_aligned_dataset import target_size  # noqa: E402


class Scheme4GeometryTests(unittest.TestCase):
    def test_scheme4_canvases_preserve_exact_ratios(self) -> None:
        expected = {
            "1280x512": (384, 960),
            "1280x704": (352, 640),
            "704x1280": (640, 352),
        }
        for bucket, canvas in expected.items():
            with self.subTest(bucket=bucket):
                height, width = canvas
                native_width, native_height = map(int, bucket.split("x"))
                self.assertEqual(scheme4_canvas(bucket), canvas)
                self.assertEqual(width * native_height, height * native_width)
                self.assertEqual(height % 32, 0)
                self.assertEqual(width % 32, 0)

    def test_video_resize_uses_the_exact_canvas_without_crop(self) -> None:
        frames = np.zeros((2, 512, 1280, 3), dtype=np.uint8)
        frames[:, :, :640] = 255

        output = resize_video_to_scheme4_canvas(
            frames,
            height=384,
            width=960,
        )

        self.assertEqual(tuple(output.shape), (3, 2, 384, 960))
        self.assertGreater(float(output[:, :, :, :480].mean()), 0.9)
        self.assertLess(float(output[:, :, :, 480:].mean()), -0.9)

    def test_reference_uses_aspect_fit_and_replicate_padding(self) -> None:
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[:, :256] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            Image.fromarray(image).save(path)
            output = resize_reference_to_scheme4_canvas(
                path,
                height=384,
                width=960,
            )

        self.assertEqual(tuple(output.shape), (3, 384, 960))
        self.assertGreater(float(output[:, :, 0].mean()), 0.9)
        self.assertLess(float(output[:, :, -1].mean()), -0.9)

    def test_unknown_or_misaligned_canvas_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            scheme4_canvas("640x640")
        frames = np.zeros((1, 512, 1280, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "32-aligned"):
            resize_video_to_scheme4_canvas(frames, height=385, width=960)

    def test_dataset_target_size_uses_scheme4_bucket_mapping(self) -> None:
        mapping = {
            "1280x512": [384, 960],
            "1280x704": [352, 640],
            "704x1280": [640, 352],
        }
        self.assertEqual(
            target_size(
                704,
                1280,
                align=16,
                canvas_by_bucket=mapping,
            ),
            (352, 640),
        )

    def test_training_entrypoint_passes_scheme4_canvas_mapping(self) -> None:
        source = (SCRIPTS / "train_teacher_aligned.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("canvas_by_bucket", source)


if __name__ == "__main__":
    unittest.main()
