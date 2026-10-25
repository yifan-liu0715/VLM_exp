from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Scheme3ControllerTests(unittest.TestCase):
    def test_launcher_is_native_guarded_and_skips_gpu_preflight(self) -> None:
        source = (
            ROOT / "scripts" / "launch_scheme3_train.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("train_scheme3_native_2000.yaml", source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)
        self.assertIn('"$GPU" = "0"', source)
        self.assertIn("21474836480", source)
        self.assertNotIn("--preflight", source)
        self.assertNotIn("--max-long-side", source)

    def test_training_entrypoint_has_no_removed_preflight_namespace_reference(self) -> None:
        source = (
            ROOT / "scripts" / "train_teacher_aligned.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("args.preflight", source)

    def test_controller_validates_then_trains_then_infers(self) -> None:
        source = (
            ROOT / "scripts" / "run_scheme3_train_then_infer.sh"
        ).read_text(encoding="utf-8")

        tests_at = source.index("unittest discover")
        cpu_at = source.index("preflight_scheme3_dataset.py")
        train_at = source.index("launch_scheme3_train.sh")
        infer_at = source.index("infer_scheme3_native.py")
        self.assertLess(tests_at, cpu_at)
        self.assertLess(cpu_at, train_at)
        self.assertLess(train_at, infer_at)
        self.assertIn("v2v_lora_and_source_patch.safetensors", source)
        self.assertNotIn("GPU preflight", source)

    def test_inference_uses_native_dataset_contract(self) -> None:
        source = (
            ROOT / "scripts" / "infer_scheme3_native.py"
        ).read_text(encoding="utf-8")

        self.assertIn("TeacherAlignedPairedDataset", source)
        self.assertIn("target_size(", source)
        self.assertNotIn("max_long_side", source)
        self.assertIn("NativeResolution_LoRA", source)
        self.assertIn("comparison_4grid", source)


if __name__ == "__main__":
    unittest.main()
