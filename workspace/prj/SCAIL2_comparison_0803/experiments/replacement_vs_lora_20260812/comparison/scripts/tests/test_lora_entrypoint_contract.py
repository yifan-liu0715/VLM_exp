import hashlib
import tempfile
import unittest
from pathlib import Path

from mode_comparison.contracts import PROMPT
from scripts.infer_scheme4_lora_only import (
    INFERENCE_VARIANTS,
    build_runtime_config,
    parse_visible_gpu,
    validate_input_manifest,
    verify_sha256,
)


class LoRAEntrypointTests(unittest.TestCase):
    def valid_manifest(self):
        return {
            "sample_id": "mymotion__humandance__batch1_mingxing_cropped__002",
            "prompt": PROMPT,
            "width": 512,
            "height": 928,
            "num_frames": 65,
            "fps": 16,
            "source_video": "/experiment/wan_lora/inputs/source.mp4",
            "target_video": "/experiment/wan_lora/inputs/target.mp4",
            "reference_image": "/experiment/wan_lora/inputs/reference.png",
            "generation_conditioning": [
                "/experiment/wan_lora/inputs/source.mp4",
                "/experiment/wan_lora/inputs/reference.png",
                PROMPT,
            ],
            "target_usage": "evaluation-only; excluded from generation conditioning",
        }

    def test_only_latest_lora_variant_is_defined(self):
        self.assertEqual(INFERENCE_VARIANTS, ("latest_lora",))

    def test_input_contract_requires_common_geometry_timing_and_no_target_leakage(self):
        validated = validate_input_manifest(self.valid_manifest(), require_files=False)

        self.assertEqual(validated["num_frames"], 65)
        self.assertEqual((validated["width"], validated["height"]), (512, 928))
        self.assertNotIn(validated["target_video"], validated["generation_conditioning"])

    def test_wrong_frame_count_is_rejected(self):
        payload = self.valid_manifest()
        payload["num_frames"] = 97
        with self.assertRaises(ValueError):
            validate_input_manifest(payload, require_files=False)

    def test_target_conditioning_is_rejected(self):
        payload = self.valid_manifest()
        payload["generation_conditioning"].append(payload["target_video"])
        with self.assertRaises(ValueError):
            validate_input_manifest(payload, require_files=False)

    def test_runtime_config_matches_scail_sampling_contract(self):
        config = build_runtime_config(self.valid_manifest())

        self.assertEqual(config["inference"]["steps"], 40)
        self.assertEqual(config["inference"]["guidance_scale"], 5.0)
        self.assertEqual(config["inference"]["flow_shift"], 3.0)
        self.assertEqual(config["inference"]["seed"], 42)

    def test_exactly_one_nonzero_physical_gpu_is_required(self):
        self.assertEqual(parse_visible_gpu("5"), "5")
        for value in ("", "0", "2,3", "GPU-abc"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_visible_gpu(value)

    def test_adapter_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            path.write_bytes(b"latest")
            expected = hashlib.sha256(b"latest").hexdigest()

            self.assertEqual(verify_sha256(path, expected), expected)
            with self.assertRaises(ValueError):
                verify_sha256(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
