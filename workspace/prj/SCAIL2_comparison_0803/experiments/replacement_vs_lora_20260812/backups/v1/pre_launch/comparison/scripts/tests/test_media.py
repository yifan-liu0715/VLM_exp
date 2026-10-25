import unittest

import numpy as np

from mode_comparison.media import (
    ACTOR_RGB,
    colorize_reference_mask,
    colorize_replacement_mask,
    fit_pad_mask,
    fit_pad_rgb,
    mask_statistics,
)


class ReplacementPaletteTests(unittest.TestCase):
    def test_reference_uses_black_background_and_actor_color(self):
        mask = np.array([[False, True]], dtype=bool)

        rendered = colorize_reference_mask(mask)

        self.assertEqual(rendered.tolist(), [[[0, 0, 0], list(ACTOR_RGB)]])

    def test_replacement_video_uses_white_background_and_same_actor_color(self):
        mask = np.array([[False, True]], dtype=bool)

        rendered = colorize_replacement_mask(mask)

        self.assertEqual(rendered.tolist(), [[[255, 255, 255], list(ACTOR_RGB)]])

    def test_colorizers_require_nonempty_boolean_mask(self):
        with self.assertRaises(ValueError):
            colorize_reference_mask(np.zeros((2, 2), dtype=bool))
        with self.assertRaises(ValueError):
            colorize_replacement_mask(np.ones((2, 2), dtype=np.uint8))


class GeometryTests(unittest.TestCase):
    def test_fit_pad_mask_is_nearest_neighbor_boolean_with_false_padding(self):
        source = np.array([[True, False], [False, False]], dtype=bool)

        output = fit_pad_mask(source, width=4, height=2)

        self.assertEqual(output.dtype, np.bool_)
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(output.tolist(), [[False, True, False, False], [False] * 4])

    def test_fit_pad_rgb_replicates_edges(self):
        source = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )

        output = fit_pad_rgb(source, width=4, height=2)

        self.assertEqual(output.shape, (2, 4, 3))
        self.assertEqual(output[0, 0].tolist(), source[0, 0].tolist())
        self.assertEqual(output[0, -1].tolist(), source[0, -1].tolist())


class MaskStatisticsTests(unittest.TestCase):
    def test_statistics_report_nonempty_fraction_and_bounds(self):
        mask = np.array([[False, True], [False, True]], dtype=bool)

        stats = mask_statistics(mask)

        self.assertEqual(stats["pixels"], 2)
        self.assertEqual(stats["fraction"], 0.5)
        self.assertEqual(stats["bbox_xyxy"], [1, 0, 2, 2])


if __name__ == "__main__":
    unittest.main()
