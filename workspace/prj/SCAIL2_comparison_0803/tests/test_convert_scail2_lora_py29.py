import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_scail2_lora_py29.py"
SPEC = importlib.util.spec_from_file_location("convert_scail2_lora_py29", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TorchLoadCompatibilityTest(unittest.TestCase):
    def test_loads_trusted_deepspeed_checkpoint_with_numpy_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dpo.pt"
            tensor = torch.ones(2, 3, dtype=torch.bfloat16)
            torch.save(
                {
                    "module": {"model.layer.lora_layer.down.weight": tensor},
                    "np_rng_state": np.random.get_state(),
                },
                path,
            )
            state = MODULE.load_module_state(path)
            self.assertEqual(list(state), ["model.layer.lora_layer.down.weight"])
            self.assertTrue(torch.equal(state["model.layer.lora_layer.down.weight"], tensor))


if __name__ == "__main__":
    unittest.main()
