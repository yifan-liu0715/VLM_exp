from __future__ import annotations

from pathlib import Path
import sys
import unittest

from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from teacher_aligned_system import (  # noqa: E402
    assert_reference_position_gradients,
    compact_trainable_state,
)
from training_contracts import validate_training_config  # noqa: E402


CONFIG = ROOT / "configs" / "train_scheme4_gate_1000.yaml"


class Scheme4AdapterContractTests(unittest.TestCase):
    def test_compact_state_includes_reference_position_tables(self) -> None:
        state = {
            "transformer.reference_position_encoder.temporal": torch.ones(1),
            "transformer.reference_position_encoder.height": torch.ones(1),
            "transformer.reference_position_encoder.width": torch.ones(1),
            "frozen.weight": torch.ones(1),
        }

        compact = compact_trainable_state(state)

        self.assertEqual(set(compact), set(state) - {"frozen.weight"})

    def test_scheme4_config_has_exact_gate_settings(self) -> None:
        config = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)

        validate_training_config(config)

        self.assertEqual(config["training"]["max_steps"], 1000)
        self.assertEqual(config["data"]["bucket_align"], 32)
        self.assertEqual(
            config["data"]["canvas_by_bucket"]["1280x704"],
            [352, 640],
        )
        self.assertEqual(config["reference_position"]["max_height"], 64)
        self.assertEqual(config["reference_position"]["max_width"], 64)
        self.assertIs(config["reference_position"]["enabled"], True)

    def test_reference_position_gradients_must_be_finite_and_nonzero(self) -> None:
        good = torch.nn.Parameter(torch.ones(2))
        good.grad = torch.tensor([0.0, 1.0])
        assert_reference_position_gradients({"temporal": good})

        missing = torch.nn.Parameter(torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, "missing"):
            assert_reference_position_gradients({"height": missing})

        zero = torch.nn.Parameter(torch.ones(2))
        zero.grad = torch.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "nonzero"):
            assert_reference_position_gradients({"width": zero})

        nonfinite = torch.nn.Parameter(torch.ones(2))
        nonfinite.grad = torch.tensor([float("nan"), 1.0])
        with self.assertRaisesRegex(RuntimeError, "finite"):
            assert_reference_position_gradients({"temporal": nonfinite})

    def test_scheme4_contract_rejects_wrong_gate_length(self) -> None:
        config = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
        config["training"]["max_steps"] = 999

        with self.assertRaisesRegex(ValueError, "1000"):
            validate_training_config(config)

    def test_training_system_installs_and_exports_reference_position(self) -> None:
        source = (SCRIPTS / "teacher_aligned_system.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("install_reference_position_encoding(", source)
        self.assertIn('"reference_position"', source)
        self.assertIn('"reference_position_encoder"', source)


if __name__ == "__main__":
    unittest.main()
