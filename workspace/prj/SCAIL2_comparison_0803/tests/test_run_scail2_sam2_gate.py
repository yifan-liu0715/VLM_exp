import unittest

from scripts.run_scail2_sam2_gate import assert_fair_pair, select_runs


class ControllerTest(unittest.TestCase):
    def test_only_ready_pair_is_selected(self):
        plan = {
            "runs": [
                {"sample_id": "dance", "variant": "base", "ready": True},
                {"sample_id": "dance", "variant": "dpo", "ready": True},
                {"sample_id": "wide", "variant": "base", "ready": False},
            ]
        }
        self.assertEqual(
            [item["variant"] for item in select_runs(plan, "dance")], ["base", "dpo"]
        )

    def test_unready_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not ready"):
            select_runs(
                {"runs": [{"sample_id": "dance", "variant": "base", "ready": False}]},
                "dance",
            )

    def test_fair_pair_only_allows_output_and_dpo_lora_differences(self):
        common = ["generate.py", "--prompt", "dance", "--base_seed", "42"]
        base = {"argv": common + ["--save_file", "/out/base.mp4"]}
        dpo = {
            "argv": common
            + [
                "--save_file",
                "/out/dpo.mp4",
                "--lora_path",
                "/weights/dpo.safetensors",
                "--lora_alpha",
                "1.0",
            ]
        }
        assert_fair_pair(base, dpo)

    def test_changed_seed_is_rejected(self):
        base = {"argv": ["generate.py", "--base_seed", "42", "--save_file", "base.mp4"]}
        dpo = {
            "argv": [
                "generate.py",
                "--base_seed",
                "7",
                "--save_file",
                "dpo.mp4",
                "--lora_path",
                "dpo.safetensors",
                "--lora_alpha",
                "1.0",
            ]
        }
        with self.assertRaisesRegex(ValueError, "fairness mismatch"):
            assert_fair_pair(base, dpo)


if __name__ == "__main__":
    unittest.main()
