from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "infer_scheme4_native_supplement.py"


class Scheme4NativeSupplementContractTests(unittest.TestCase):
    def test_uses_scheme3_native_geometry_and_position_interpolation(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            "load_aligned_sample as load_native_aligned_sample",
            source,
        )
        self.assertIn(
            "interpolate_reference_position_for_native_grid(",
            source,
        )
        self.assertNotIn("resize_video_to_scheme4_canvas", source)
        self.assertNotIn("resize_reference_to_scheme4_canvas", source)

    def test_records_exact_scheme3_conditions_and_interpolation(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        for marker in (
            '"num_frames": int(native_cfg.data.num_frames)',
            '"steps": int(native_cfg.inference.steps)',
            '"guidance_scale": float(native_cfg.inference.guidance_scale)',
            '"seed": int(native_cfg.inference.seed)',
            '"reference_position_interpolation": interpolation_report',
            '"reused_scheme3_native_baseline": True',
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
