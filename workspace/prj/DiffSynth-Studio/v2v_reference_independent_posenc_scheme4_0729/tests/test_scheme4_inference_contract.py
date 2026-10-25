from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from infer_scheme4 import (  # noqa: E402
    adapter_tensor_contract,
    validate_adapter_contract,
)


def state_with_contract(
    *,
    lora: int,
    source_patch: int,
    reference_position: int,
) -> dict[str, torch.Tensor]:
    state = {
        **{f"transformer.block.{index}.lora_A": torch.ones(1) for index in range(lora)},
        **{
            f"transformer.patch_embedding_extra.{index}": torch.ones(1)
            for index in range(source_patch)
        },
        **{
            f"transformer.reference_position_encoder.table_{index}": torch.ones(1)
            for index in range(reference_position)
        },
    }
    return state


class Scheme4InferenceContractTests(unittest.TestCase):
    def test_scheme4_requires_exact_reference_position_tables(self) -> None:
        state = state_with_contract(
            lora=480,
            source_patch=2,
            reference_position=3,
        )
        self.assertEqual(
            adapter_tensor_contract(state),
            {
                "lora": 480,
                "source_patch": 2,
                "reference_position": 3,
                "other": 0,
            },
        )
        validate_adapter_contract(state, require_reference_position=True)
        missing = state_with_contract(
            lora=480,
            source_patch=2,
            reference_position=0,
        )
        with self.assertRaisesRegex(RuntimeError, "reference_position"):
            validate_adapter_contract(
                missing,
                require_reference_position=True,
            )

    def test_scheme3_contract_forbids_reference_position_tables(self) -> None:
        state = state_with_contract(
            lora=480,
            source_patch=2,
            reference_position=0,
        )
        validate_adapter_contract(state, require_reference_position=False)

    def test_script_controls_position_encoder_and_complete_outputs(self) -> None:
        source = (SCRIPTS / "infer_scheme4.py").read_text(encoding="utf-8")
        self.assertIn("reference_position_encoding_enabled = False", source)
        self.assertIn("reference_position_encoding_enabled = True", source)
        for filename in (
            "Source.mp4",
            "Target.mp4",
            "Base.mp4",
            "Scheme3.mp4",
            "Scheme4.mp4",
            "comparison_base_scheme4_4grid.mp4",
            "comparison_scheme3_scheme4_4grid.mp4",
            "run_manifest.json",
        ):
            self.assertIn(filename, source)
        self.assertIn("shared_initial_noise", source)

    def test_gate_controller_runs_tests_preflight_train_then_infer(self) -> None:
        source = (SCRIPTS / "run_scheme4_gate.sh").read_text(encoding="utf-8")
        order = [
            source.index("unittest discover"),
            source.index("preflight_scheme4.py"),
            source.index("launch_scheme4_train.sh"),
            source.index("infer_scheme4.py"),
        ]
        self.assertEqual(order, sorted(order))
        self.assertIn('"$GPU" = "0"', source)
        self.assertIn("21474836480", source)
        audit_at = source.index("audit_scheme4_results.py")
        complete_at = source.index("touch \"$ROOT/validation/COMPLETE\"")
        self.assertLess(order[-1], audit_at)
        self.assertLess(audit_at, complete_at)


if __name__ == "__main__":
    unittest.main()
