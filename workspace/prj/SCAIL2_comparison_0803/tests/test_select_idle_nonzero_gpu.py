import unittest

from scripts.select_idle_nonzero_gpu import select_gpu


class SelectGpuTest(unittest.TestCase):
    def test_idle_gpu_zero_is_never_selected(self):
        rows = [
            {"index": 0, "memory_free_mib": 90000, "utilization_gpu": 0},
            {"index": 2, "memory_free_mib": 80000, "utilization_gpu": 2},
            {"index": 5, "memory_free_mib": 70000, "utilization_gpu": 0},
        ]
        self.assertEqual(select_gpu(rows, 65000), 2)

    def test_busy_gpu_is_rejected(self):
        rows = [
            {"index": 1, "memory_free_mib": 90000, "utilization_gpu": 40},
            {"index": 3, "memory_free_mib": 68000, "utilization_gpu": 10},
        ]
        self.assertEqual(select_gpu(rows, 65000), 3)

    def test_no_eligible_gpu_fails(self):
        with self.assertRaisesRegex(RuntimeError, "no eligible nonzero GPU"):
            select_gpu(
                [{"index": 0, "memory_free_mib": 90000, "utilization_gpu": 0}],
                1000,
            )


if __name__ == "__main__":
    unittest.main()
