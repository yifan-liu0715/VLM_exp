from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_teacher_aligned_manifests import build_manifests, hash_tree


FIELDNAMES = [
    "sample_id",
    "bucket",
    "width",
    "height",
    "num_frames",
    "input_video",
    "output_video",
    "reference_image",
    "caption",
    "source_split",
]


def write_fixture(path: Path) -> None:
    rows = [
        {
            "sample_id": "mymotion__002",
            "bucket": "1280x704",
            "width": "1280",
            "height": "704",
            "num_frames": "118",
            "input_video": "/dataset/1280x704/output_video/mymotion__002.mp4",
            "output_video": "/dataset/1280x704/input_video/mymotion__002.mp4",
            "reference_image": "/dataset/1280x704/reference/mymotion__002.jpg",
            "caption": "target caption",
            "source_split": "train",
        },
        {
            "sample_id": "heldout_val",
            "bucket": "704x1280",
            "width": "704",
            "height": "1280",
            "num_frames": "120",
            "input_video": "/dataset/704x1280/output_video/heldout_val.mp4",
            "output_video": "/dataset/704x1280/input_video/heldout_val.mp4",
            "reference_image": "/dataset/704x1280/reference/heldout_val.jpg",
            "caption": "validation caption",
            "source_split": "val",
        },
        {
            "sample_id": "heldout_test",
            "bucket": "1280x512",
            "width": "1280",
            "height": "512",
            "num_frames": "121",
            "input_video": "/dataset/1280x512/output_video/heldout_test.mp4",
            "output_video": "/dataset/1280x512/input_video/heldout_test.mp4",
            "reference_image": "/dataset/1280x512/reference/heldout_test.jpg",
            "caption": "test caption",
            "source_split": "test",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class ManifestContractTests(unittest.TestCase):
    def test_reference_hash_tree_is_relative_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            (root / "models" / "a.py").write_text("alpha\n", encoding="utf-8")
            (root / "models" / "__pycache__").mkdir()
            (root / "models" / "__pycache__" / "a.pyc").write_bytes(b"cache")
            (root / "README.md").write_text("teacher\n", encoding="utf-8")

            first = hash_tree(root)
            second = hash_tree(root)

            self.assertEqual(first, second)
            self.assertEqual(set(first), {"README.md", "models/a.py"})
            self.assertTrue(all(len(digest) == 64 for digest in first.values()))

    def test_split_manifests_are_disjoint_and_train_is_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            write_fixture(source)

            report = build_manifests(source, root / "metadata")

            train_path = root / "metadata" / "train_manifest.csv"
            with train_path.open(encoding="utf-8") as handle:
                train_rows = list(csv.DictReader(handle))
            self.assertEqual({row["source_split"] for row in train_rows}, {"train"})
            self.assertEqual(report["split_counts"], {"train": 1, "val": 1, "test": 1})

            ids_by_split = {}
            for split in ("train", "val", "test"):
                with (root / "metadata" / f"{split}_manifest.csv").open(
                    encoding="utf-8"
                ) as handle:
                    rows = list(csv.DictReader(handle))
                ids_by_split[split] = {row["sample_id"] for row in rows}
            self.assertFalse(ids_by_split["train"] & ids_by_split["val"])
            self.assertFalse(ids_by_split["train"] & ids_by_split["test"])
            self.assertFalse(ids_by_split["val"] & ids_by_split["test"])

    def test_known_sample_records_physical_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            write_fixture(source)

            report = build_manifests(source, root / "metadata")
            sample = report["known_samples"]["mymotion__002"]

            self.assertIn("/output_video/", sample["source_video"])
            self.assertIn("/input_video/", sample["target_video"])
            self.assertEqual(
                report["contract"],
                "physical output_video + reference + caption -> physical input_video",
            )

            persisted = json.loads(
                (root / "metadata" / "manifest_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, report)

    def test_unknown_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            write_fixture(source)
            with source.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["source_split"] = "unknown"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "unsupported source_split"):
                build_manifests(source, root / "metadata")


if __name__ == "__main__":
    unittest.main()
