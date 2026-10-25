from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from teacher_aligned_dataset import (
    aligned_frame_indices,
    load_manifest_rows,
    target_size,
)


class TeacherAlignedDatasetContractTests(unittest.TestCase):
    def test_aligned_indices_cover_exactly_97_frames(self) -> None:
        indices = aligned_frame_indices(
            paired_length=118,
            sample_frames=97,
            start=7,
        )
        self.assertEqual(indices[0], 7)
        self.assertEqual(indices[-1], 103)
        self.assertEqual(len(indices), 97)

    def test_aligned_indices_reject_short_or_out_of_range_clips(self) -> None:
        with self.assertRaisesRegex(ValueError, "shorter"):
            aligned_frame_indices(paired_length=96, sample_frames=97, start=0)
        with self.assertRaisesRegex(ValueError, "start"):
            aligned_frame_indices(paired_length=118, sample_frames=97, start=22)

    def test_teacher_bucket_size_preserves_ratio_and_alignment(self) -> None:
        self.assertEqual(
            target_size(704, 1280, max_long_side=1088, align=32),
            (576, 1088),
        )
        self.assertEqual(
            target_size(1280, 704, max_long_side=1088, align=32),
            (1088, 576),
        )
        self.assertEqual(
            target_size(512, 1280, max_long_side=1088, align=32),
            (416, 1088),
        )

    def test_loader_rejects_non_train_rows_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            fieldnames = [
                "sample_id",
                "bucket",
                "width",
                "height",
                "num_frames",
                "input_video",
                "output_video",
                "reference_image",
                "caption",
                "source_split",
            ]
            row = {
                "sample_id": "bad",
                "bucket": "1280x704",
                "width": "1280",
                "height": "704",
                "num_frames": "118",
                "input_video": "/dataset/output_video/bad.mp4",
                "output_video": "/dataset/input_video/bad.mp4",
                "reference_image": "/dataset/reference/bad.jpg",
                "caption": "caption",
                "source_split": "val",
            }
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "non-train"):
                load_manifest_rows(path, required_split="train", check_media=False)


if __name__ == "__main__":
    unittest.main()
