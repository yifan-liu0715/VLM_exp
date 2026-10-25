import unittest

import numpy as np

from sam2_fallback_0808.detector import (
    Detection,
    multiclass_nms,
    select_single_or_manual,
)


class DetectorTest(unittest.TestCase):
    def test_single_person_is_selected(self):
        actor = Detection((200, 100, 600, 1100), 0.88, 0)
        self.assertEqual(select_single_or_manual([actor], (704, 1280)), actor)

    def test_multiple_people_require_mask_iou_matching(self):
        people = [
            Detection((0, 0, 100, 100), 0.95, 0),
            Detection((200, 100, 600, 1100), 0.88, 0),
        ]
        with self.assertRaisesRegex(ValueError, "mask IoU"):
            select_single_or_manual(people, (704, 1280))

    def test_manual_box_is_bounded_and_recorded(self):
        chosen = select_single_or_manual([], (704, 1280), (100, 50, 600, 1200))
        self.assertEqual(chosen.box_xyxy, (100.0, 50.0, 600.0, 1200.0))
        self.assertEqual(chosen.class_id, -1)
        with self.assertRaisesRegex(ValueError, "outside"):
            select_single_or_manual([], (704, 1280), (-1, 50, 600, 1200))

    def test_no_person_fails(self):
        with self.assertRaisesRegex(ValueError, "no person"):
            select_single_or_manual([], (704, 1280))

    def test_nms_suppresses_overlapping_lower_score_box(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100]], dtype=np.float32)
        scores = np.array([[0.9], [0.7]], dtype=np.float32)
        kept = multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.3)
        self.assertEqual(kept.shape[0], 1)
        self.assertAlmostEqual(float(kept[0, 4]), 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
