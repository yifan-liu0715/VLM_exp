import unittest

from mode_comparison.contracts import (
    DANCE_SAMPLE_ID,
    LATEST_LORA_SHA256,
    SCAIL2_SHA256,
    canvas_for,
    comparison_contract,
    temporal_indices,
)


class TemporalIndexTests(unittest.TestCase):
    def test_full_duration_indices_span_all_97_source_frames(self):
        indices = temporal_indices(97, 65)

        self.assertEqual(len(indices), 65)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 96)
        self.assertEqual(tuple(sorted(set(indices))), indices)
        self.assertEqual(indices[1], 2)
        self.assertEqual(indices[3], 5)

    def test_temporal_indices_reject_invalid_counts(self):
        for input_frames, output_frames in ((0, 65), (97, 0), (1, 2), (64, 65)):
            with self.subTest(input_frames=input_frames, output_frames=output_frames):
                with self.assertRaises(ValueError):
                    temporal_indices(input_frames, output_frames)


class CanvasTests(unittest.TestCase):
    def test_dance_canvas_is_32_aligned_512_by_928(self):
        width, height = canvas_for(704, 1280)

        self.assertEqual((width, height), (512, 928))
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)

    def test_unknown_canvas_is_rejected(self):
        with self.assertRaises(ValueError):
            canvas_for(1920, 1080)


class FrozenContractTests(unittest.TestCase):
    def test_replacement_comparison_contract_is_frozen(self):
        contract = comparison_contract("replacement")

        self.assertEqual(contract.mode, "replacement")
        self.assertEqual(contract.sample_id, DANCE_SAMPLE_ID)
        self.assertEqual((contract.width, contract.height), (512, 928))
        self.assertEqual((contract.input_frames, contract.input_fps), (97, 24))
        self.assertEqual((contract.output_frames, contract.output_fps), (65, 16))
        self.assertEqual(contract.steps, 40)
        self.assertEqual(contract.guidance_scale, 5.0)
        self.assertEqual(contract.shift, 3.0)
        self.assertEqual(contract.seed, 42)
        self.assertEqual(contract.scail2_sha256, SCAIL2_SHA256)
        self.assertEqual(contract.lora_sha256, LATEST_LORA_SHA256)

    def test_only_supported_modes_are_accepted(self):
        comparison_contract("animation")
        with self.assertRaises(ValueError):
            comparison_contract("unknown")


if __name__ == "__main__":
    unittest.main()
