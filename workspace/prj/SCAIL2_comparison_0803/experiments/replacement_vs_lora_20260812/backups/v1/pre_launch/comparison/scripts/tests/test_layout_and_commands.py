import tempfile
import unittest
from pathlib import Path
import json

from mode_comparison.commands import (
    build_lora_command,
    build_scail_command,
    fairness_contract,
)
from mode_comparison.contracts import comparison_contract
from mode_comparison.layout import (
    directory_relpaths,
    initialize_layout,
    weight_pointer_records,
)
from scripts.build_mode_experiment import build_experiment


class LayoutTests(unittest.TestCase):
    def test_each_model_has_separate_artifact_directories(self):
        relpaths = directory_relpaths()

        self.assertEqual(len(relpaths), len(set(relpaths)))
        for branch in ("scail2", "wan_lora"):
            for artifact in ("scripts", "weights", "inputs", "outputs", "logs", "manifests"):
                self.assertIn(f"{branch}/{artifact}", relpaths)
        self.assertIn("comparison/outputs", relpaths)
        self.assertIn("backups/v1", relpaths)

    def test_layout_refuses_nonempty_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            root.mkdir()
            (root / "existing.txt").write_text("old", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                initialize_layout(root)

    def test_weight_records_are_pointers_with_frozen_hashes(self):
        contract = comparison_contract("replacement")

        records = weight_pointer_records(contract)

        self.assertEqual(records["scail2"]["path"], contract.scail2_weight)
        self.assertEqual(records["scail2"]["sha256"], contract.scail2_sha256)
        self.assertEqual(records["wan_lora"]["path"], contract.lora_weight)
        self.assertEqual(records["wan_lora"]["sha256"], contract.lora_sha256)
        self.assertEqual(records["policy"], "pointer_only_no_weight_copy")

    def test_builder_writes_small_pointer_and_plan_manifests_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "replacement_vs_lora_20260812"

            result = build_experiment(root, "replacement")

            scail_pointer = json.loads(
                (root / "scail2/weights/weights.json").read_text(encoding="utf-8")
            )
            lora_pointer = json.loads(
                (root / "wan_lora/weights/weights.json").read_text(encoding="utf-8")
            )
            plan = json.loads(
                (root / "comparison/manifests/run_plan.json").read_text(encoding="utf-8")
            )
        self.assertEqual(result["status"], "awaiting_inputs")
        self.assertEqual(scail_pointer["policy"], "pointer_only_no_weight_copy")
        self.assertIn("SCAIL-2.safetensors", scail_pointer["weight"]["path"])
        self.assertIn("final_adapter", lora_pointer["weight"]["path"])
        self.assertEqual(plan["mode"], "replacement")
        self.assertIn("--replace_flag", plan["scail2"]["argv"])


class CommandTests(unittest.TestCase):
    def test_replacement_scail_command_has_replace_flag_and_isolated_output(self):
        contract = comparison_contract("replacement")
        command = build_scail_command(contract)

        self.assertIn("--replace_flag", command)
        self.assertIn("--segment_len", command)
        self.assertIn("65", command)
        self.assertTrue(command[-1].endswith("scail2/outputs/scail2_replacement.mp4"))

    def test_animation_scail_command_has_no_replace_flag(self):
        command = build_scail_command(comparison_contract("animation"))

        self.assertNotIn("--replace_flag", command)
        self.assertTrue(command[-1].endswith("scail2/outputs/scail2_animation.mp4"))

    def test_lora_command_runs_only_latest_adapter_output(self):
        contract = comparison_contract("replacement")
        command = build_lora_command(contract)
        joined = " ".join(command).lower()

        self.assertIn(contract.lora_weight, command)
        self.assertIn(contract.lora_sha256, command)
        self.assertNotIn("scheme3", joined)
        self.assertNotIn("base.mp4", joined)
        self.assertTrue(command[-1].endswith("wan_lora/manifests/run_manifest.json"))

    def test_fairness_contract_freezes_common_generation_parameters(self):
        contract = fairness_contract(comparison_contract("replacement"))

        self.assertEqual(contract["width"], 512)
        self.assertEqual(contract["height"], 928)
        self.assertEqual(contract["num_frames"], 65)
        self.assertEqual(contract["fps"], 16)
        self.assertEqual(contract["steps"], 40)
        self.assertEqual(contract["guidance_scale"], 5.0)
        self.assertEqual(contract["shift"], 3.0)
        self.assertEqual(contract["seed"], 42)
        self.assertEqual(len(contract["prompt_sha256"]), 64)
        self.assertEqual(contract["shared_input_sha256"], "pending_preprocessing")


if __name__ == "__main__":
    unittest.main()
