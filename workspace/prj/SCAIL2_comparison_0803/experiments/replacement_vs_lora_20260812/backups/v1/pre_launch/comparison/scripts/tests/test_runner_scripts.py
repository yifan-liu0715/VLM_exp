import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunnerScriptTests(unittest.TestCase):
    def read(self, name: str) -> str:
        return (ROOT / "scripts" / name).read_text(encoding="utf-8")

    def test_scail_runner_is_replacement_only_and_gpu0_safe(self):
        text = self.read("run_scail2_replacement.sh")

        self.assertIn("set -euo pipefail", text)
        self.assertIn('if [ "$GPU" = "0" ]', text)
        self.assertIn('export CUDA_VISIBLE_DEVICES="$GPU"', text)
        self.assertIn("--replace_flag", text)
        self.assertIn("scail2/logs/run.log", text)
        self.assertIn("scail2/outputs/scail2_replacement.mp4", text)
        self.assertNotIn("scail2_animation.mp4", text)

    def test_lora_runner_runs_only_latest_adapter_and_its_own_log(self):
        text = self.read("run_lora_replacement.sh")
        lowered = text.lower()

        self.assertIn("set -euo pipefail", text)
        self.assertIn('if [ "$GPU" = "0" ]', text)
        self.assertIn('export CUDA_VISIBLE_DEVICES="$GPU"', text)
        self.assertIn("dece6fe3aa1de90166f861c4e10405ab32ceb4b1a78f61e9bac08af5f595908a", text)
        self.assertIn("wan_lora/logs/run.log", text)
        self.assertIn("wan_lora/outputs/lora.mp4", text)
        self.assertNotIn("scheme3", lowered)
        self.assertNotIn("base.mp4", lowered)

    def test_both_runners_refuse_busy_or_low_memory_gpu(self):
        for name in ("run_scail2_replacement.sh", "run_lora_replacement.sh"):
            with self.subTest(name=name):
                text = self.read(name)
                self.assertIn("--query-compute-apps=pid", text)
                self.assertIn("--query-gpu=memory.free", text)
                self.assertIn("70000", text)


if __name__ == "__main__":
    unittest.main()
