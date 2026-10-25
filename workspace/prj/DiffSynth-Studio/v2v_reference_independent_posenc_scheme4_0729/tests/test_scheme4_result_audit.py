from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_scheme4_results import (  # noqa: E402
    REQUIRED_VIDEO_NAMES,
    expected_sample_outputs,
    validate_video_probe,
)


class Scheme4ResultAuditTests(unittest.TestCase):
    def test_expected_outputs_include_seven_videos_and_manifest(self) -> None:
        outputs = expected_sample_outputs(Path("/results"), "sample")

        self.assertEqual(set(outputs["videos"]), set(REQUIRED_VIDEO_NAMES))
        self.assertEqual(len(outputs["videos"]), 7)
        self.assertEqual(
            outputs["manifest"],
            Path("/results/sample/run_manifest.json"),
        )

    def test_probe_requires_exact_frame_count_and_rate(self) -> None:
        valid = {
            "streams": [
                {
                    "codec_type": "video",
                    "nb_read_frames": "97",
                    "avg_frame_rate": "24/1",
                    "width": 640,
                    "height": 352,
                }
            ]
        }
        validate_video_probe(valid)

        wrong_frames = {
            "streams": [
                {
                    "codec_type": "video",
                    "nb_read_frames": "96",
                    "avg_frame_rate": "24/1",
                    "width": 640,
                    "height": 352,
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "97"):
            validate_video_probe(wrong_frames)

        wrong_rate = {
            "streams": [
                {
                    "codec_type": "video",
                    "nb_read_frames": "97",
                    "avg_frame_rate": "25/1",
                    "width": 640,
                    "height": 352,
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "24"):
            validate_video_probe(wrong_rate)


if __name__ == "__main__":
    unittest.main()
