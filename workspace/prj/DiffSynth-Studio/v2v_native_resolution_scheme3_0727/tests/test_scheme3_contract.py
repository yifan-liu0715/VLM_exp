from __future__ import annotations

from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from training_contracts import validate_training_config


class Scheme3ContractTests(unittest.TestCase):
    @property
    def config_path(self) -> Path:
        return ROOT / "configs" / "train_scheme3_native_2000.yaml"

    def test_config_uses_native_resolution(self) -> None:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["training"]["run_kind"], "native_resolution_scheme3")
        self.assertTrue(config["data"]["native_resolution"])
        self.assertNotIn("max_long_side", config["data"])
        self.assertEqual(config["data"]["num_frames"], 97)
        self.assertEqual(config["training"]["max_steps"], 2000)

    def test_scheme3_contract_accepts_exact_settings(self) -> None:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        validate_training_config(config)

    def test_scheme3_contract_rejects_spatial_limit(self) -> None:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        config["data"]["max_long_side"] = 704

        with self.assertRaisesRegex(ValueError, "native"):
            validate_training_config(config)

    def test_scheme3_contract_rejects_disabled_native_resolution(self) -> None:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        config["data"]["native_resolution"] = False

        with self.assertRaisesRegex(ValueError, "native_resolution"):
            validate_training_config(config)


if __name__ == "__main__":
    unittest.main()
