import hashlib
import tempfile
import unittest
from pathlib import Path

from sam2_fallback_0808.assets import AssetSpec, sha256_file, verify_asset


class AssetTest(unittest.TestCase):
    def test_hash_and_expected_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset.bin"
            path.write_bytes(b"official")
            digest = hashlib.sha256(b"official").hexdigest()
            spec = AssetSpec("fixture", "https://example.invalid/a", str(path), digest)

            self.assertEqual(sha256_file(path), digest)
            self.assertTrue(verify_asset(path, spec)["verified"])

            bad = AssetSpec("fixture", spec.url, str(path), "0" * 64)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_asset(path, bad)

    def test_missing_asset_is_rejected(self):
        spec = AssetSpec("missing", "https://example.invalid/a", "/missing", None)
        with self.assertRaises(FileNotFoundError):
            verify_asset(Path(spec.path), spec)


if __name__ == "__main__":
    unittest.main()
