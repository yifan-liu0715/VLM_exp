from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL_PATCH = ROOT / "model_patch"
for path in (SCRIPTS, MODEL_PATCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.wan2.reference_position_encoding import (  # noqa: E402
    FactorizedReferencePositionEncoding,
)
from native_reference_position import (  # noqa: E402
    interpolate_reference_position_for_native_grid,
)


class NativeReferencePositionTests(unittest.TestCase):
    def test_interpolates_only_used_spatial_prefixes_to_native_grid(self) -> None:
        module = FactorizedReferencePositionEncoding(
            dim=2,
            max_frames=4,
            max_height=64,
            max_width=64,
            init_std=0.02,
        )
        with torch.no_grad():
            module.temporal.copy_(
                torch.arange(8, dtype=torch.float32).reshape(4, 2)
            )
            module.height.copy_(
                torch.arange(128, dtype=torch.float32).reshape(64, 2)
            )
            module.width.copy_(
                torch.arange(128, dtype=torch.float32).reshape(64, 2) * 10
            )
        original_temporal = module.temporal.detach().clone()
        original_height_endpoints = (
            module.height[0].detach().clone(),
            module.height[23].detach().clone(),
        )
        original_width_endpoints = (
            module.width[0].detach().clone(),
            module.width[59].detach().clone(),
        )

        report = interpolate_reference_position_for_native_grid(
            module,
            trained_grid=(24, 60),
            native_grid=(32, 80),
        )

        self.assertEqual(tuple(module.height.shape), (32, 2))
        self.assertEqual(tuple(module.width.shape), (80, 2))
        torch.testing.assert_close(module.temporal, original_temporal)
        torch.testing.assert_close(module.height[0], original_height_endpoints[0])
        torch.testing.assert_close(
            module.height[-1], original_height_endpoints[1]
        )
        torch.testing.assert_close(module.width[0], original_width_endpoints[0])
        torch.testing.assert_close(module.width[-1], original_width_endpoints[1])
        self.assertEqual(report["trained_grid"], [24, 60])
        self.assertEqual(report["native_grid"], [32, 80])
        self.assertEqual(report["mode"], "linear_align_corners")

    def test_rejects_nonpositive_grid_dimensions(self) -> None:
        module = FactorizedReferencePositionEncoding(
            dim=2,
            max_frames=1,
            max_height=4,
            max_width=4,
            init_std=0.02,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            interpolate_reference_position_for_native_grid(
                module,
                trained_grid=(2, 2),
                native_grid=(0, 4),
            )


if __name__ == "__main__":
    unittest.main()
