import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam2_fallback_0808.palette import encode_reference, encode_video_frame
from sam2_fallback_0808.quality import compute_mask_stats
from scripts import validate_sam2_masks as validator


class PaletteQualityTest(unittest.TestCase):
    def test_exact_colors(self):
        mask = np.array([[False, True]], bool)
        np.testing.assert_array_equal(encode_reference(mask)[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(encode_video_frame(mask)[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(encode_reference(mask)[0, 1], [0, 0, 255])
        np.testing.assert_array_equal(encode_video_frame(mask)[0, 1], [0, 0, 255])

    def test_area_jump_fails_automatic_gate(self):
        masks = [np.zeros((10, 10), bool) for _ in range(3)]
        masks[0][:2, :2] = True
        masks[1][:8, :8] = True
        masks[2][:8, :8] = True
        report = compute_mask_stats(masks)
        self.assertEqual(report["jump_frames"], [1])
        self.assertFalse(report["automatic_pass"])

    def test_empty_mask_is_reported(self):
        masks = [np.ones((4, 4), bool), np.zeros((4, 4), bool)]
        report = compute_mask_stats(masks)
        self.assertEqual(report["empty_frames"], [1])
        self.assertFalse(report["automatic_pass"])

    def test_shape_change_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mask shape mismatch"):
            compute_mask_stats([np.ones((4, 4), bool), np.ones((4, 5), bool)])

    def test_driving_video_is_trimmed_to_manifest_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            for index in range(5):
                image = np.full((16, 16, 3), index * 40, dtype=np.uint8)
                Image.fromarray(image).save(frames / f"frame_{index:05d}.png")
            source = root / "source.mp4"
            output = root / "prefix.mp4"
            validator._encode_png_sequence(frames, 24.0, source)

            validator._encode_video_prefix(source, frame_count=3, fps=24.0, output=output)

            self.assertEqual(validator._frame_count(validator._ffprobe(output)), 3)


if __name__ == "__main__":
    unittest.main()
