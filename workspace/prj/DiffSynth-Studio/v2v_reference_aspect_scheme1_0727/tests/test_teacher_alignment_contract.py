from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from training_contracts import (
    explicit_nonzero_visible_gpus,
    scaled_learning_rate,
    teacher_lr_multiplier,
    validate_training_config,
)


class TeacherAlignmentContractTests(unittest.TestCase):
    def test_config_records_teacher_aligned_recommended_settings(self) -> None:
        text = (
            ROOT / "configs" / "train_teacher_aligned_20000.yaml"
        ).read_text(encoding="utf-8")
        required_fragments = [
            "max_steps: 20000",
            "num_frames: 97",
            "rank: 96",
            "alpha: 96",
            "beta2: 0.95",
            "warmup_steps: 2",
            "text_drop_prob: 0.1",
            "reference_drop_prob: 0.5",
            "flow_shift: 5.0",
            "seed: 1234",
            "max_long_side: 704",
            "accumulate_grad_batches: 1",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        self.assertNotIn("quick", text.lower())
        self.assertNotIn("conservative", text.lower())

    def test_teacher_lr_scaling_matches_reference_formula(self) -> None:
        self.assertAlmostEqual(
            scaled_learning_rate(
                base_lr=1e-4,
                accumulate_grad_batches=4,
                num_gpus=1,
                num_nodes=1,
            ),
            2e-4,
        )
        self.assertAlmostEqual(
            scaled_learning_rate(
                base_lr=1e-4,
                accumulate_grad_batches=4,
                num_gpus=3,
                num_nodes=1,
            ),
            1e-4 * math.sqrt(12),
        )

    def test_teacher_lr_multiplier_warms_up_then_stays_constant(self) -> None:
        self.assertEqual(teacher_lr_multiplier(step=0, warmup_steps=2), 0.0)
        self.assertEqual(teacher_lr_multiplier(step=1, warmup_steps=2), 0.5)
        self.assertEqual(teacher_lr_multiplier(step=2, warmup_steps=2), 1.0)
        self.assertEqual(teacher_lr_multiplier(step=20_000, warmup_steps=2), 1.0)

    def test_gpu_guard_rejects_gpu0_duplicates_and_more_than_three(self) -> None:
        for value in ("", "0", "0,2", "2,2", "1,2,4,6", "one"):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    explicit_nonzero_visible_gpus(value)
        self.assertEqual(
            explicit_nonzero_visible_gpus("1,2,4"),
            ["1", "2", "4"],
        )

    def test_formal_config_guard_rejects_wrong_step_or_split(self) -> None:
        valid = {
            "data": {
                "train_manifest": "/experiment/metadata/train_manifest.csv",
                "num_frames": 97,
                "max_long_side": 1088,
            },
            "lora": {"rank": 96, "alpha": 96},
            "training": {
                "max_steps": 20_000,
                "text_drop_prob": 0.1,
                "reference_drop_prob": 0.5,
            },
            "output": {"root": "/experiment"},
        }
        validate_training_config(valid)
        invalid_steps = {
            **valid,
            "training": {**valid["training"], "max_steps": 8_000},
        }
        with self.assertRaisesRegex(ValueError, "20000"):
            validate_training_config(invalid_steps)
        invalid_manifest = {
            **valid,
            "data": {**valid["data"], "train_manifest": "/metadata/all.csv"},
        }
        with self.assertRaisesRegex(ValueError, "train_manifest.csv"):
            validate_training_config(invalid_manifest)

    def test_launcher_source_has_gpu0_and_disk_guards(self) -> None:
        source = (SCRIPTS / "launch_train.sh").read_text(encoding="utf-8")
        self.assertIn("GPU0 is forbidden", source)
        self.assertIn("MIN_FREE_KB", source)
        self.assertIn("diffsynth_fixed/bin/python", source)
        self.assertIn("v2v_formal_0715/environment/python_packages", source)
        self.assertIn("train_teacher_aligned.py", source)

    def test_training_writes_start_manifest_before_fit(self) -> None:
        source = (SCRIPTS / "train_teacher_aligned.py").read_text(encoding="utf-8")
        self.assertIn("run_start_manifest.json", source)
        self.assertIn("teacher_reference_sha256", source)
        self.assertIn("effective_global_batch", source)
        self.assertLess(
            source.index("run_start_manifest.json"),
            source.index("trainer.fit("),
        )

    def test_training_has_overwrite_recovery_checkpoint(self) -> None:
        trainer_source = (SCRIPTS / "train_teacher_aligned.py").read_text(
            encoding="utf-8"
        )
        system_source = (SCRIPTS / "teacher_aligned_system.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("recovery.ckpt", trainer_source)
        self.assertIn("trainer.save_checkpoint", trainer_source)
        self.assertIn("ckpt_path=recovery_path", trainer_source)
        self.assertIn("def on_save_checkpoint", system_source)
        self.assertIn('checkpoint["state_dict"] = compact_trainable_state', system_source)

    def test_system_source_preserves_teacher_flow_matching_contract(self) -> None:
        source = (SCRIPTS / "teacher_aligned_system.py").read_text(encoding="utf-8")
        required_fragments = [
            "hidden_states=noisy_target",
            '"encoder_contion_states": source',
            '"encoder_first_states": reference',
            "velocity_target = self.flow_scheduler.training_target(target, noise, timestep)",
            "text_drop_prob",
            "reference_drop_prob",
            'target_modules=["to_q", "to_k", "to_v", "to_out.0"]',
            "materialize_vae_statistics",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
