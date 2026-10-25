from __future__ import annotations

import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tiered_training_utils import (  # noqa: E402
    MILESTONES,
    milestone_adapter_path,
    swap_v2v_direction,
    validate_milestones,
)


class CorrectedDirectionTests(unittest.TestCase):
    def test_swaps_only_source_and_target_media(self) -> None:
        row = {
            "sample_id": "mymotion__002",
            "input_video": "/dataset/input_video/mymotion__002.mp4",
            "output_video": "/dataset/output_video/mymotion__002.mp4",
            "reference_image": "/dataset/reference/mymotion__002.jpg",
            "caption": "Two women sit in green armchairs.",
            "bucket": "1280x704",
        }

        corrected = swap_v2v_direction(row)

        self.assertEqual(corrected["input_video"], row["output_video"])
        self.assertEqual(corrected["output_video"], row["input_video"])
        self.assertEqual(corrected["reference_image"], row["reference_image"])
        self.assertEqual(corrected["caption"], row["caption"])
        self.assertEqual(corrected["sample_id"], row["sample_id"])
        self.assertEqual(row["input_video"], "/dataset/input_video/mymotion__002.mp4")

    def test_rejects_identical_source_and_target(self) -> None:
        row = {
            "input_video": "/dataset/same.mp4",
            "output_video": "/dataset/same.mp4",
        }
        with self.assertRaisesRegex(ValueError, "must differ"):
            swap_v2v_direction(row)


class MilestoneTests(unittest.TestCase):
    def test_canonical_milestones_cover_all_three_tiers(self) -> None:
        validate_milestones(MILESTONES, max_steps=20_000)
        self.assertEqual(
            MILESTONES,
            {3265: "quick", 6530: "conservative", 20000: "recommended"},
        )

    def test_final_milestone_must_equal_max_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "final milestone"):
            validate_milestones({3265: "quick", 6530: "conservative"}, max_steps=20_000)

    def test_tier_names_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_milestones({3265: "quick", 6530: "quick"}, max_steps=6530)

    def test_adapter_path_is_isolated_by_tier(self) -> None:
        path = milestone_adapter_path(Path("/run"), 6530, MILESTONES)
        self.assertEqual(
            path,
            Path("/run/conservative/final_adapter/v2v_lora_and_source_patch.safetensors"),
        )

    def test_unknown_milestone_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "not configured"):
            milestone_adapter_path(Path("/run"), 999, MILESTONES)


if __name__ == "__main__":
    unittest.main()
