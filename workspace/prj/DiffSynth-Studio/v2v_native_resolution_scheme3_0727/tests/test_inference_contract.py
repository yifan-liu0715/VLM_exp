from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "infer_scheme2_flowfix.py"
CONFIG = ROOT / "configs" / "train_scheme2_2000.yaml"


class TeacherAlignedInferenceContractTests(unittest.TestCase):
    def test_config_preserves_teacher_inference_defaults(self) -> None:
        source = CONFIG.read_text(encoding="utf-8")
        self.assertIn("steps: 50", source)
        self.assertIn("guidance_scale: 5.0", source)
        self.assertIn("flow_shift: 5.0", source)
        self.assertIn("seed: 42", source)

    def test_inference_uses_unipc_with_teacher_flow_shift(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("UniPCMultistepScheduler(", source)
        self.assertIn('prediction_type="flow_prediction"', source)
        self.assertIn("flow_shift=float(cfg.inference.flow_shift)", source)
        self.assertIn("scheduler.set_timesteps(steps, device=latents.device)", source)

    def test_base_and_adapter_share_noise_and_conditions(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("initial_noise.clone()", source)
        self.assertIn("conditional_attention_kwargs", source)
        self.assertIn("unconditional_attention_kwargs = conditional_attention_kwargs", source)
        main = source.index("def main")
        base = source.index('label="base"', main)
        adapter_load = source.index("load_teacher_adapter(", base)
        adapter = source.index('label="adapter"', adapter_load)
        self.assertLess(
            base,
            adapter_load,
        )
        self.assertGreater(
            adapter,
            adapter_load,
        )

    def test_cfg_runs_conditional_and_unconditional_text_branches(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("conditional_prediction = transformer(", source)
        self.assertIn("unconditional_prediction = transformer(", source)
        normalized = " ".join(source.split())
        self.assertIn(
            "unconditional_prediction + guidance_scale * (",
            normalized,
        )
        self.assertIn(
            "conditional_prediction - unconditional_prediction",
            normalized,
        )

    def test_outputs_are_auditable_and_only_final_adapter_is_default(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("recommended/final_adapter", source)
        self.assertIn('"source"', source)
        self.assertIn('"target"', source)
        self.assertIn('"base"', source)
        self.assertIn('"adapter"', source)
        self.assertIn('"comparison"', source)
        self.assertIn("run_manifest.json", source)
        self.assertNotIn("quick", source)
        self.assertNotIn("conservative", source)
        self.assertIn(
            "nearest_align32_aspect_fit_replicate_minimum_padding",
            source,
        )
        self.assertIn("validate_paired_frame_shapes", source)
        self.assertIn("REFERENCE + VIDEO GEOMETRY SCHEME 2", source)


if __name__ == "__main__":
    unittest.main()
