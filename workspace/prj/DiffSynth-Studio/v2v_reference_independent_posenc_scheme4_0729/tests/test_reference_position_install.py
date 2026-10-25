from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "model_patch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from install_reference_position_encoding import (  # noqa: E402
    install_reference_position_encoding,
)


class DummyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv3d(4, 8, kernel_size=(1, 2, 2))
        self.reference_position_encoder = None
        self.reference_position_encoding_enabled = False


class ReferencePositionInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimpleNamespace(
            reference_position=SimpleNamespace(
                max_frames=4,
                max_height=64,
                max_width=64,
                init_std=0.02,
            )
        )

    def test_install_uses_transformer_inner_dimension(self) -> None:
        transformer = DummyTransformer()

        module = install_reference_position_encoding(
            transformer,
            self.cfg,
            trainable=True,
            enabled=True,
        )

        self.assertIs(module, transformer.reference_position_encoder)
        self.assertEqual(module.temporal.shape, (4, 8))
        self.assertEqual(module.height.shape, (64, 8))
        self.assertEqual(module.width.shape, (64, 8))
        self.assertTrue(transformer.reference_position_encoding_enabled)
        self.assertTrue(all(p.requires_grad for p in module.parameters()))

    def test_install_can_disable_and_freeze_for_base_inference(self) -> None:
        transformer = DummyTransformer()

        module = install_reference_position_encoding(
            transformer,
            self.cfg,
            trainable=False,
            enabled=False,
        )

        self.assertFalse(transformer.reference_position_encoding_enabled)
        self.assertFalse(any(p.requires_grad for p in module.parameters()))


if __name__ == "__main__":
    unittest.main()
