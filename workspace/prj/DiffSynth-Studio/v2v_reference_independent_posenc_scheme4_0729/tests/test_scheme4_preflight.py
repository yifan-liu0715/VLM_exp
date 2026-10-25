from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from preflight_scheme4 import latent_and_patch_shapes  # noqa: E402


class Scheme4PreflightTests(unittest.TestCase):
    def test_latent_and_patch_shapes_fit_reference_tables(self) -> None:
        expected = {
            (384, 960): {
                "video_latent": [48, 25, 24, 60],
                "reference_grid": [1, 12, 30],
            },
            (352, 640): {
                "video_latent": [48, 25, 22, 40],
                "reference_grid": [1, 11, 20],
            },
            (640, 352): {
                "video_latent": [48, 25, 40, 22],
                "reference_grid": [1, 20, 11],
            },
        }
        for canvas, wanted in expected.items():
            with self.subTest(canvas=canvas):
                observed = latent_and_patch_shapes(
                    height=canvas[0],
                    width=canvas[1],
                    num_frames=97,
                )
                self.assertEqual(observed["video_latent"], wanted["video_latent"])
                self.assertEqual(observed["reference_grid"], wanted["reference_grid"])
                self.assertLessEqual(observed["reference_grid"][0], 4)
                self.assertLessEqual(observed["reference_grid"][1], 64)
                self.assertLessEqual(observed["reference_grid"][2], 64)


if __name__ == "__main__":
    unittest.main()
