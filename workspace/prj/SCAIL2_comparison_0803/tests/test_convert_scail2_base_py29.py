import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_scail2_base_py29.py"
SPEC = importlib.util.spec_from_file_location("convert_scail2_base_py29", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BaseConversionCompatibilityTest(unittest.TestCase):
    def test_loads_trusted_checkpoint_with_mmap_compatible_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "base.pt"
            tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            torch.save(
                {
                    "module": {"model.diffusion_model.time_embed.0.weight": tensor},
                    "np_rng_state": np.random.get_state(),
                },
                path,
            )
            state = MODULE.load_module_state(path)
            self.assertEqual(list(state), ["model.diffusion_model.time_embed.0.weight"])
            self.assertTrue(torch.equal(state["model.diffusion_model.time_embed.0.weight"], tensor))


if __name__ == "__main__":
    unittest.main()
