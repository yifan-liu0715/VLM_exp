import unittest

from scripts.build_scail2_sam2_report import validate_pair_metadata


def pair_manifest(*, seed="42", label="official SAM2 fallback; not official SAM3"):
    common = [
        "/env/python",
        "generate.py",
        "--image",
        "/conditions/ref.jpg",
        "--mask_image",
        "/conditions/ref_mask.jpg",
        "--pose",
        "/conditions/driving.mp4",
        "--mask_video",
        "/conditions/mask.mp4",
        "--base_seed",
        "42",
    ]
    base = common + ["--save_file", "/out/base.mp4"]
    dpo = common[:-1] + [seed, "--save_file", "/out/dpo.mp4", "--lora_path", "/weights/dpo.safetensors", "--lora_alpha", "1.0"]
    return {
        "mask_preprocessor": label,
        "fairness_check": "passed",
        "runs": [
            {"variant": "base", "command": base},
            {"variant": "dpo", "command": dpo},
        ],
    }


class ReportValidationTest(unittest.TestCase):
    def test_accepts_only_dpo_lora_difference(self):
        validate_pair_metadata(
            pair_manifest(),
            {"base": {"reference": "abc"}, "dpo": {"reference": "abc"}},
        )

    def test_changed_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fairness mismatch"):
            validate_pair_metadata(
                pair_manifest(seed="7"),
                {"base": {"reference": "abc"}, "dpo": {"reference": "abc"}},
            )

    def test_condition_hash_difference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "condition hashes"):
            validate_pair_metadata(
                pair_manifest(),
                {"base": {"reference": "abc"}, "dpo": {"reference": "def"}},
            )

    def test_missing_sam2_fallback_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SAM2 fallback"):
            validate_pair_metadata(
                pair_manifest(label="unknown"),
                {"base": {"reference": "abc"}, "dpo": {"reference": "abc"}},
            )


if __name__ == "__main__":
    unittest.main()
