from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tiered_training_utils.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tiered_training_utils", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TieredTrainingTests(unittest.TestCase):
    def test_exact_milestones_map_to_isolated_adapter_paths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = module.validate_milestones(
                {6530: "conservative", 20000: "recommended"}, max_steps=20000
            )
            self.assertEqual(
                module.milestone_adapter_path(root, mapping, 6530),
                root / "conservative/final_adapter/v2v_lora_and_source_patch.safetensors",
            )
            self.assertEqual(
                module.milestone_adapter_path(root, mapping, 20000),
                root / "recommended/final_adapter/v2v_lora_and_source_patch.safetensors",
            )

    def test_last_milestone_must_equal_max_steps(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "last milestone"):
            module.validate_milestones(
                {6530: "conservative", 19999: "recommended"}, max_steps=20000
            )

    def test_milestones_require_unique_labels_and_positive_steps(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "positive"):
            module.validate_milestones(
                {0: "conservative", 20000: "recommended"}, max_steps=20000
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            module.validate_milestones(
                {6530: "same", 20000: "same"}, max_steps=20000
            )


if __name__ == "__main__":
    unittest.main()
