import tempfile
import unittest
from pathlib import Path

from mode_comparison.contracts import (
    MASK_PREPROCESSOR,
    REFERENCE_IMAGE,
    SOURCE_VIDEO,
    TARGET_VIDEO,
    comparison_contract,
)
from scripts.prepare_mode_inputs import (
    build_frame_index_record,
    prepared_paths,
    validate_raw_mask_manifest,
    validate_video_metadata,
)


class FrameIndexRecordTests(unittest.TestCase):
    def test_record_covers_full_motion_at_16_fps(self):
        record = build_frame_index_record(comparison_contract("replacement"))

        self.assertEqual(record["source_frames"], 97)
        self.assertEqual(record["source_fps"], 24)
        self.assertEqual(record["output_frames"], 65)
        self.assertEqual(record["output_fps"], 16)
        self.assertEqual(record["indices"][0], 0)
        self.assertEqual(record["indices"][-1], 96)
        self.assertEqual(len(record["samples"]), 65)
        self.assertEqual(record["samples"][-1]["output_time_seconds"], 4.0)
        self.assertEqual(record["samples"][-1]["source_time_seconds"], 4.0)

    def test_120_frame_dataset_file_is_valid_for_first_97_frame_clip(self):
        result = validate_video_metadata(
            observed_frames=120,
            observed_fps=24.0,
            observed_width=704,
            observed_height=1280,
            contract=comparison_contract("replacement"),
        )

        self.assertEqual(result["file_frames"], 120)
        self.assertEqual(result["clip_frames"], 97)
        self.assertEqual(result["clip_start"], 0)
        self.assertEqual(result["clip_end_inclusive"], 96)

    def test_video_shorter_than_97_frame_clip_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_video_metadata(
                observed_frames=96,
                observed_fps=24.0,
                observed_width=704,
                observed_height=1280,
                contract=comparison_contract("replacement"),
            )


class MaskLineageTests(unittest.TestCase):
    def valid_manifest(self):
        return {
            "sample_id": comparison_contract("replacement").sample_id,
            "mask_preprocessor": "official SAM2 fallback; not official SAM3",
            "conditioning_paths": [SOURCE_VIDEO, REFERENCE_IMAGE],
            "frame_count": 97,
            "fps": 24.0,
            "width": 704,
            "height": 1280,
        }

    def test_valid_sam2_lineage_is_normalized_to_explicit_fallback_label(self):
        result = validate_raw_mask_manifest(
            self.valid_manifest(), comparison_contract("replacement")
        )

        self.assertEqual(result["mask_preprocessor"], MASK_PREPROCESSOR)
        self.assertNotIn(TARGET_VIDEO, result["conditioning_paths"])

    def test_target_conditioning_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["conditioning_paths"].append(TARGET_VIDEO)

        with self.assertRaises(ValueError):
            validate_raw_mask_manifest(manifest, comparison_contract("replacement"))


class PreparedPathTests(unittest.TestCase):
    def test_model_and_comparison_inputs_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = prepared_paths(Path(directory))

        all_paths = [str(path) for path in paths.values()]
        self.assertEqual(len(all_paths), len(set(all_paths)))
        self.assertIn("scail2", str(paths["scail_driving"]))
        self.assertIn("wan_lora", str(paths["wan_source"]))
        self.assertIn("comparison", str(paths["comparison_source"]))
        self.assertTrue(str(paths["scail_replace_mask"]).endswith("replace_mask.mp4"))


if __name__ == "__main__":
    unittest.main()
