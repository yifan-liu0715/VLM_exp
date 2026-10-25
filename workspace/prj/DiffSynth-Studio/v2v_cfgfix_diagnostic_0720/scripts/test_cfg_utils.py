from __future__ import annotations

import importlib
import sys
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class ClassifierFreeGuidanceTests(unittest.TestCase):
    def _helper(self):
        try:
            module = importlib.import_module("cfg_utils")
        except ModuleNotFoundError:
            self.fail("cfg_utils.apply_classifier_free_guidance is missing")
        return module.apply_classifier_free_guidance

    def test_combines_conditional_and_unconditional_predictions(self):
        apply_cfg = self._helper()
        self.assertEqual(apply_cfg(conditional=3.0, unconditional=1.0, guidance_scale=5.0), 11.0)

    def test_requires_guidance_scale_above_one(self):
        apply_cfg = self._helper()
        with self.assertRaisesRegex(ValueError, "greater than 1"):
            apply_cfg(conditional=3.0, unconditional=1.0, guidance_scale=1.0)


if __name__ == "__main__":
    unittest.main()

