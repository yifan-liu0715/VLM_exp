from __future__ import annotations

import ast
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("infer_v2v_comparison_cfg.py")


class InferenceCfgContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_cli_defaults_to_teacher_guidance_scale(self):
        self.assertIn('"--guidance-scale"', self.source)
        self.assertIn("default=5.0", self.source)

    def test_denoising_computes_unconditional_prediction_and_cfg(self):
        self.assertIn("negative_prompt_embeddings", self.source)
        self.assertIn("unconditional_prediction", self.source)
        self.assertIn("apply_classifier_free_guidance", self.source)

    def test_manifest_truthfully_records_cfg(self):
        self.assertIn('"cfg_enabled": True', self.source)
        self.assertIn('"guidance_scale": args.guidance_scale', self.source)


if __name__ == "__main__":
    unittest.main()

