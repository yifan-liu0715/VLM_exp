from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATCH = ROOT / "model_patch"
if str(MODEL_PATCH) not in sys.path:
    sys.path.insert(0, str(MODEL_PATCH))

from models.wan2.reference_position_encoding import (  # noqa: E402
    FactorizedReferencePositionEncoding,
)


class ReferencePositionEncodingTests(unittest.TestCase):
    def test_transformer_applies_reference_encoding_only_when_enabled(self) -> None:
        source = (
            MODEL_PATCH / "models" / "wan2" / "transformer_wan.py"
        ).read_text(encoding="utf-8")
        normalized = "".join(source.split())

        self.assertIn("self.reference_position_encoder = None", source)
        self.assertIn("self.reference_position_encoding_enabled = False", source)
        self.assertIn(
            "reference_position = self.reference_position_encoder(",
            source,
        )
        self.assertIn(
            "encoder_first_states+reference_position",
            normalized,
        )

    def test_reference_encoding_is_spatially_varying(self) -> None:
        torch.manual_seed(7)
        module = FactorizedReferencePositionEncoding(
            dim=8,
            max_frames=2,
            max_height=4,
            max_width=4,
            init_std=0.02,
        )
        patched = torch.zeros(1, 8, 1, 3, 3)

        output = module(patched)

        self.assertEqual(tuple(output.shape), (1, 9, 8))
        self.assertFalse(torch.equal(output[:, 0], output[:, 1]))
        differences = output - output[:, :1]
        self.assertGreater(torch.count_nonzero(differences).item(), 0)
        self.assertGreater(
            torch.unique(differences.reshape(-1, 8), dim=0).shape[0],
            2,
        )

    def test_reference_encoding_is_deterministic_for_fixed_state(self) -> None:
        torch.manual_seed(11)
        module = FactorizedReferencePositionEncoding(
            dim=8,
            max_frames=2,
            max_height=4,
            max_width=4,
            init_std=0.02,
        )
        patched = torch.zeros(2, 8, 1, 2, 3)

        first = module(patched)
        second = module(patched)

        torch.testing.assert_close(first, second)

    def test_reference_encoding_rejects_out_of_bounds_grid(self) -> None:
        module = FactorizedReferencePositionEncoding(
            dim=8,
            max_frames=1,
            max_height=2,
            max_width=2,
            init_std=0.02,
        )

        with self.assertRaisesRegex(ValueError, "exceeds"):
            module(torch.zeros(1, 8, 1, 3, 2))

    def test_reference_position_tables_receive_nonzero_gradients(self) -> None:
        torch.manual_seed(19)
        module = FactorizedReferencePositionEncoding(
            dim=8,
            max_frames=2,
            max_height=4,
            max_width=4,
            init_std=0.02,
        )
        patched = torch.zeros(1, 8, 1, 3, 3)

        module(patched).square().mean().backward()

        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)


if __name__ == "__main__":
    unittest.main()
