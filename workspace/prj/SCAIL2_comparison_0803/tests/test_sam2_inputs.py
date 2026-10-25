import json
import tempfile
import unittest
from pathlib import Path

from sam2_fallback_0808.inputs import assert_conditioning_paths, load_sample


class InputBoundaryTest(unittest.TestCase):
    def test_source_reference_mapping_and_target_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            manifest = {
                "samples": [
                    {
                        "sample_id": "dance",
                        "source_video": "/data/output_video/dance.mp4",
                        "target_video": "/data/input_video/dance.mp4",
                        "reference_image": "/data/reference/dance.jpg",
                        "prompt": "dance",
                        "width": 704,
                        "height": 1280,
                        "num_frames": 97,
                        "fps": 24.0,
                    }
                ]
            }
            (root / "manifests" / "input_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            sample = load_sample(root, "dance")
            self.assertIn("output_video", str(sample.source_video))
            self.assertIn("input_video", str(sample.target_video))
            assert_conditioning_paths(sample, [sample.source_video, sample.reference_image])
            with self.assertRaisesRegex(ValueError, "target leakage"):
                assert_conditioning_paths(sample, [sample.target_video])

    def test_reversed_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            manifest = {
                "samples": [
                    {
                        "sample_id": "bad",
                        "source_video": "/data/input_video/bad.mp4",
                        "target_video": "/data/output_video/bad.mp4",
                        "reference_image": "/data/reference/bad.jpg",
                        "prompt": "bad",
                        "width": 1280,
                        "height": 512,
                        "num_frames": 97,
                        "fps": 24.0,
                    }
                ]
            }
            (root / "manifests" / "input_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "approved dataset direction"):
                load_sample(root, "bad")


if __name__ == "__main__":
    unittest.main()
