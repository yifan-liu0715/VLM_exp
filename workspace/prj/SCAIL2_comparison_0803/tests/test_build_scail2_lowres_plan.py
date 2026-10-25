import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_scail2_lowres_plan.py"
SPEC = importlib.util.spec_from_file_location("lowres_plan", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def option_map(argv):
    result = {}
    index = 1
    while index < len(argv):
        key = argv[index]
        if key == "--replace_flag":
            result[key] = True
            index += 1
        else:
            result[key] = argv[index + 1]
            index += 2
    return result


class LowResolutionPlanTest(unittest.TestCase):
    def test_lowres_plan_changes_only_canvas_and_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir(parents=True)
            (root / "manifests" / "input_manifest.json").write_text(
                json.dumps({"samples": [{"sample_id": "dance", "prompt": "dance", "width": 704, "height": 1280}]}),
                encoding="utf-8",
            )
            processed = root / "preprocessed_sam2" / "dance"
            processed.mkdir(parents=True)
            for name in ("ref_mask.jpg", "rendered_v2.mp4", "rendered_mask_v2.mp4"):
                (processed / name).write_bytes(b"fixture")
            (processed / "mask_manifest.json").write_text(
                json.dumps({"sample_id": "dance", "status": "passed", "mask_preprocessor": "official SAM2 fallback; not official SAM3"}),
                encoding="utf-8",
            )

            plan = MODULE.build_lowres_plan(root, segment_len=65)
            base = next(run for run in plan["runs"] if run["variant"] == "base")
            dpo = next(run for run in plan["runs"] if run["variant"] == "dpo")
            base_opts = option_map(base["argv"])
            dpo_opts = option_map(dpo["argv"])

            self.assertEqual(plan["resolution_policy"], "lowres_32_aligned_aspect_preserving")
            self.assertEqual(base_opts["--target_w"], "512")
            self.assertEqual(base_opts["--target_h"], "928")
            self.assertEqual(base_opts["--segment_len"], "65")
            self.assertIn("sam2_lowres_base_vs_dpo", base_opts["--save_file"])
            self.assertEqual(base_opts["--target_w"], dpo_opts["--target_w"])
            self.assertEqual(base_opts["--target_h"], dpo_opts["--target_h"])
            self.assertEqual(base_opts["--segment_len"], dpo_opts["--segment_len"])


if __name__ == "__main__":
    unittest.main()
