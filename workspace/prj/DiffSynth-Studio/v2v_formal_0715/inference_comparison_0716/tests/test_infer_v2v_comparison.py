from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "infer_v2v_comparison.py"


def load_module():
    spec = importlib.util.spec_from_file_location("infer_v2v_comparison", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InferComparisonTests(unittest.TestCase):
    def test_training_bucket_size_is_height_then_width(self):
        module = load_module()
        self.assertEqual(module.target_size_for_bucket("1280x704"), (320, 576))

    def test_adapter_keys_drop_only_transformer_prefix(self):
        module = load_module()
        key = "transformer.blocks.0.attn1.to_q.lora_A.default.weight"
        self.assertEqual(
            module.normalize_adapter_key(key),
            "blocks.0.attn1.to_q.lora_A.default.weight",
        )
        self.assertEqual(module.normalize_adapter_key("other.weight"), "other.weight")

    def test_adapter_key_summary_separates_lora_and_source_patch(self):
        module = load_module()
        keys = [
            "transformer.blocks.0.attn1.to_q.lora_A.default.weight",
            "transformer.blocks.0.attn1.to_q.lora_B.default.weight",
            "transformer.patch_embedding_extra.weight",
            "transformer.patch_embedding_extra.bias",
        ]
        self.assertEqual(
            module.summarize_adapter_keys(keys),
            {"lora": 2, "source_patch": 2, "other": 0},
        )

    def test_seeded_noise_is_reproducible_and_independently_cloned(self):
        module = load_module()
        first = module.make_seeded_noise((1, 2, 3, 4, 5), seed=42, device="cpu")
        second = module.make_seeded_noise((1, 2, 3, 4, 5), seed=42, device="cpu")
        self.assertTrue(torch.equal(first, second))
        first.add_(1)
        self.assertFalse(torch.equal(first, second))

    def test_four_panel_is_two_by_two_with_expected_geometry(self):
        module = load_module()
        source = np.zeros((2, 320, 576, 3), dtype=np.uint8)
        target = np.full_like(source, 64)
        base = np.full_like(source, 128)
        lora = np.full_like(source, 255)
        panels = module.make_four_panel(source, target, base, lora)
        self.assertEqual(panels.shape, (2, 640, 1152, 3))
        self.assertEqual(panels.dtype, np.uint8)
        self.assertEqual(panels[0, 200, 200].max(), 0)
        self.assertEqual(panels[0, 200, 800].max(), 64)
        self.assertEqual(panels[0, 500, 200].max(), 128)
        self.assertEqual(panels[0, 500, 800].max(), 255)

    def test_formal_adapter_requires_exact_tensor_classes(self):
        module = load_module()
        module.validate_adapter_summary({"lora": 480, "source_patch": 2, "other": 0})
        with self.assertRaisesRegex(ValueError, "unexpected formal adapter tensor summary"):
            module.validate_adapter_summary({"lora": 478, "source_patch": 2, "other": 0})

    def test_load_manifest_row_selects_exact_sample(self):
        module = load_module()
        csv_text = (
            "sample_id,bucket,input_video,output_video,reference_image,caption,num_frames\n"
            "other,1280x704,/a.mp4,/b.mp4,/c.jpg,other prompt,40\n"
            "mymotion__002,1280x704,/s.mp4,/t.mp4,/r.jpg,target prompt,118\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(csv_text, encoding="utf-8")
            row = module.load_manifest_row(path, "mymotion__002", require_files=False)
        self.assertEqual(row["caption"], "target prompt")
        self.assertEqual(row["bucket"], "1280x704")

    def test_tensor_to_uint8_frames_changes_bcthw_to_fhwc(self):
        module = load_module()
        video = torch.tensor(
            [[[[[-1.0, 0.0], [1.0, -1.0]]], [[[0.0, 1.0], [-1.0, 0.0]]], [[[1.0, -1.0], [0.0, 1.0]]]]]
        )
        frames = module.tensor_to_uint8_frames(video)
        self.assertEqual(frames.shape, (1, 2, 2, 3))
        self.assertEqual(frames.dtype, np.uint8)
        self.assertEqual(frames[0, 0, 0].tolist(), [0, 128, 255])


if __name__ == "__main__":
    unittest.main()
