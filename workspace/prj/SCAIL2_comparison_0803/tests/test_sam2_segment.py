import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam2_fallback_0808.segment import (
    segment_reference,
    select_source_track,
    validate_propagated_masks,
)
from scripts.prepare_sam2_masks import _read_rgb


class FakePredictor:
    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, box, multimask_output):
        masks = np.zeros((1, *self.shape), dtype=bool)
        x1, y1, x2, y2 = box[0].astype(int)
        masks[0, y1:y2, x1:x2] = True
        return masks, np.array([0.9]), np.array([0.0])


class SegmentTest(unittest.TestCase):
    def test_reference_shape_and_dtype(self):
        mask = segment_reference(
            FakePredictor(), np.zeros((128, 64, 3), np.uint8), (5, 10, 50, 100)
        )
        self.assertEqual(mask.shape, (128, 64))
        self.assertEqual(mask.dtype, np.bool_)

    def test_all_97_frames_are_required(self):
        masks = [np.ones((8, 8), bool) for _ in range(97)]
        validate_propagated_masks(masks, 97, (8, 8))
        masks[48][:] = False
        with self.assertRaisesRegex(ValueError, "empty mask at frame 48"):
            validate_propagated_masks(masks, 97, (8, 8))

    def test_multi_person_track_uses_highest_reference_iou(self):
        reference = np.zeros((8, 8), bool)
        reference[2:6, 2:6] = True
        wrong = np.zeros((8, 8), bool)
        wrong[:2, :2] = True
        right = reference.copy()
        index, scores = select_source_track(reference, [wrong, right])
        self.assertEqual(index, 1)
        self.assertGreater(scores[1], scores[0])

    def test_track_selection_rejects_empty_candidates(self):
        with self.assertRaisesRegex(ValueError, "no source track candidates"):
            select_source_track(np.ones((4, 4), bool), [])

    def test_reference_array_is_writable_for_torchvision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
            self.assertTrue(_read_rgb(path).flags.writeable)


if __name__ == "__main__":
    unittest.main()
