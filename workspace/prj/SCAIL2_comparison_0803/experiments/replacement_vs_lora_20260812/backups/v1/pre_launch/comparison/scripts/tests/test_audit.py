import unittest

from mode_comparison.audit import (
    compare_fairness_fields,
    normalize_probe,
    parse_physical_gpu,
    scan_log_errors,
)


class ProbeTests(unittest.TestCase):
    def test_probe_is_normalized_to_exact_fields(self):
        stream = {
            "width": 512,
            "height": 928,
            "avg_frame_rate": "16/1",
            "nb_read_frames": "65",
        }

        self.assertEqual(
            normalize_probe(stream),
            {"width": 512, "height": 928, "fps": 16.0, "frames": 65},
        )


class LogTests(unittest.TestCase):
    def test_clean_log_has_no_errors(self):
        self.assertEqual(scan_log_errors("Mode: Replacement\nFinished."), [])

    def test_failure_signatures_are_reported(self):
        text = "CUDA out of memory\nTraceback (most recent call last)\nnon-finite latents"

        errors = scan_log_errors(text)

        self.assertIn("cuda_oom", errors)
        self.assertIn("traceback", errors)
        self.assertIn("non_finite", errors)

    def test_physical_gpu0_is_rejected(self):
        self.assertEqual(parse_physical_gpu("physical_gpu=5"), "5")
        with self.assertRaises(ValueError):
            parse_physical_gpu("physical_gpu=0")


class FairnessTests(unittest.TestCase):
    def test_common_fields_match(self):
        scail = {
            "prompt": "same",
            "width": 512,
            "height": 928,
            "num_frames": 65,
            "fps": 16,
            "steps": 40,
            "guidance_scale": 5.0,
            "shift": 3.0,
            "seed": 42,
        }
        lora = dict(scail)

        self.assertEqual(compare_fairness_fields(scail, lora), {})

    def test_difference_is_reported_by_field(self):
        scail = {"fps": 16, "seed": 42}
        lora = {"fps": 24, "seed": 42}

        self.assertEqual(
            compare_fairness_fields(scail, lora),
            {"fps": {"scail2": 16, "wan_lora": 24}},
        )


if __name__ == "__main__":
    unittest.main()
