import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/fetch_sam2_fallback_assets.py"
SPEC = importlib.util.spec_from_file_location("fetch_sam2_assets", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FetcherRepositoryTest(unittest.TestCase):
    def test_verified_existing_commit_skips_network_fetch(self):
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repo"
            path.mkdir()

            def fake_run(argv):
                if argv[-2:] == ["rev-parse", "HEAD"]:
                    return commit
                raise AssertionError(f"unexpected network or checkout command: {argv}")

            with patch.object(MODULE, "run_checked", side_effect=fake_run):
                result = MODULE.ensure_repository(
                    path, "https://example.invalid/repo.git", commit, verify_only=False
                )
            self.assertTrue(result["verified"])
            self.assertEqual(result["commit"], commit)


if __name__ == "__main__":
    unittest.main()
