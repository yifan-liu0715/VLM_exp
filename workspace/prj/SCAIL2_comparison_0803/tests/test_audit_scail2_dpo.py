import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_scail2_dpo.py"
SPEC = importlib.util.spec_from_file_location("audit_scail2_dpo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DpoAuditTest(unittest.TestCase):
    def test_summarizes_checkpoint_on_meta_device(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            torch.save(
                {
                    "module": {
                        "layer.matrix_A": torch.zeros(4, 8, dtype=torch.bfloat16),
                        "layer.matrix_B": torch.zeros(8, 4, dtype=torch.bfloat16),
                    },
                    "iteration": 1,
                },
                path,
            )
            result = MODULE.summarize(path)
            self.assertTrue(result["all_tensors_meta"])
            self.assertEqual(result["tensor_count"], 2)
            self.assertEqual(result["total_numel"], 64)
            self.assertEqual(result["dtypes"], {"torch.bfloat16": 2})
            self.assertEqual(result["top_level_keys"], ["iteration", "module"])


if __name__ == "__main__":
    unittest.main()
