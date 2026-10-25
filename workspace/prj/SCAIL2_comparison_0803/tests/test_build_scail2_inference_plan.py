import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_scail2_inference_plan.py"
SPEC = importlib.util.spec_from_file_location("inference_plan", MODULE_PATH)
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


def write_fixture(root: Path, status: str = "passed") -> dict:
    (root / "manifests").mkdir(parents=True)
    sample = {
        "sample_id": "dance",
        "prompt": "dance prompt",
        "width": 704,
        "height": 1280,
    }
    (root / "manifests/input_manifest.json").write_text(
        json.dumps({"samples": [sample]}), encoding="utf-8"
    )
    processed = root / "preprocessed_sam2/dance"
    processed.mkdir(parents=True)
    for name in ("ref_mask.jpg", "rendered_v2.mp4", "rendered_mask_v2.mp4"):
        (processed / name).write_bytes(b"fixture")
    (processed / "mask_manifest.json").write_text(
        json.dumps(
            {
                "sample_id": "dance",
                "status": status,
                "mask_preprocessor": "official SAM2 fallback; not official SAM3",
            }
        ),
        encoding="utf-8",
    )
    return sample


class InferencePlanTest(unittest.TestCase):
    def test_passed_sam2_bundle_is_ready_and_base_dpo_are_equal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = write_fixture(root)
            plan = MODULE.build_plan(root)

            self.assertEqual(plan["mask_preprocessor"], "official SAM2 fallback; not official SAM3")
            self.assertEqual(plan["status"], "ready")
            self.assertEqual(len(plan["runs"]), 2)
            base = next(run for run in plan["runs"] if run["variant"] == "base")
            dpo = next(run for run in plan["runs"] if run["variant"] == "dpo")
            self.assertTrue(base["ready"])
            self.assertTrue(dpo["ready"])
            base_opts = option_map(base["argv"])
            dpo_opts = option_map(dpo["argv"])
            self.assertIn("preprocessed_sam2", base_opts["--mask_image"])
            self.assertEqual(base_opts["--mask_video"], dpo_opts["--mask_video"])
            self.assertNotIn("--lora_path", base_opts)
            self.assertNotIn("--lora_alpha", base_opts)
            self.assertTrue(dpo_opts.pop("--lora_path").endswith("bias-aware-dpo-lora.safetensors"))
            self.assertEqual(dpo_opts.pop("--lora_alpha"), "1.0")
            self.assertNotEqual(base_opts.pop("--save_file"), dpo_opts.pop("--save_file"))
            self.assertEqual(base_opts, dpo_opts)
            self.assertEqual(base_opts["--target_w"], str(sample["width"]))
            self.assertEqual(base_opts["--target_h"], str(sample["height"]))
            self.assertEqual(base_opts["--sample_steps"], "40")
            self.assertEqual(base_opts["--sample_guide_scale"], "5.0")
            self.assertEqual(base_opts["--base_seed"], "42")
            self.assertTrue(base_opts["--replace_flag"])

    def test_failed_manifest_keeps_both_variants_unready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, status="failed")
            plan = MODULE.build_plan(root)
            self.assertEqual(plan["status"], "blocked_on_approved_sam2_masks")
            self.assertTrue(all(not run["ready"] for run in plan["runs"]))

    def test_segment_len_override_is_identical_for_base_and_dpo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)

            plan = MODULE.build_plan(root, segment_len=65)

            base = next(run for run in plan["runs"] if run["variant"] == "base")
            dpo = next(run for run in plan["runs"] if run["variant"] == "dpo")
            self.assertEqual(option_map(base["argv"])["--segment_len"], "65")
            self.assertEqual(option_map(dpo["argv"])["--segment_len"], "65")
            self.assertEqual(plan["segment_len"], 65)

    def test_missing_manifest_is_not_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)
            (root / "preprocessed_sam2/dance/mask_manifest.json").unlink()
            self.assertFalse(MODULE.approved_mask_bundle(root, "dance"))


if __name__ == "__main__":
    unittest.main()
