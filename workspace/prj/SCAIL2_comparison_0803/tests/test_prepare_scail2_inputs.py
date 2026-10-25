import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_scail2_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_scail2_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareInputsTest(unittest.TestCase):
    def test_fixed_mapping_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_root = root / "scheme4"
            dataset = root / "dataset"
            output_root = root / "inputs"
            for index, sample_id in enumerate(MODULE.SAMPLE_IDS):
                resolution = f"r{index}"
                files = {
                    "source_video": dataset / resolution / "output_video" / f"{sample_id}.mp4",
                    "target_video": dataset / resolution / "input_video" / f"{sample_id}.mp4",
                    "reference_image": dataset / resolution / "reference" / f"{sample_id}.jpg",
                }
                for key, path in files.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{sample_id}:{key}".encode())
                manifest = {
                    "sample_id": sample_id,
                    "split": "test",
                    "prompt": f"prompt {index}",
                    "width": 1280,
                    "height": 512,
                    "num_frames": 97,
                    "fps": 24.0,
                    **{key: str(path) for key, path in files.items()},
                }
                path = manifest_root / sample_id / "run_manifest.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(manifest), encoding="utf-8")

            result = MODULE.prepare(manifest_root, output_root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["samples"]), 3)
            for record in result["samples"]:
                sample_dir = Path(record["input_dir"])
                self.assertTrue((sample_dir / "driving.mp4").is_symlink())
                self.assertTrue((sample_dir / "target.mp4").is_symlink())
                self.assertTrue((sample_dir / "ref.jpg").is_symlink())
                self.assertIn("/output_video/", record["source_video"].replace("\\", "/"))
                self.assertIn("/input_video/", record["target_video"].replace("\\", "/"))
                self.assertIn("/reference/", record["reference_image"].replace("\\", "/"))
                self.assertNotEqual(record["source_sha256"], record["target_sha256"])

    def test_rejects_reversed_source_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wrong = root / "input_video" / "x.mp4"
            target = root / "output_video" / "x.mp4"
            reference = root / "reference" / "x.jpg"
            for path in (wrong, target, reference):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "source_video"):
                MODULE.validate_mapping(
                    {
                        "sample_id": "x",
                        "source_video": str(wrong),
                        "target_video": str(target),
                        "reference_image": str(reference),
                    },
                    "x",
                )


if __name__ == "__main__":
    unittest.main()
