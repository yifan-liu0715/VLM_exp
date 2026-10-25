#!/usr/bin/env python3
"""Create deterministic source-group-disjoint train/val/test metadata splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
SOURCE_METADATA = Path("/nvmedata/workspace2/users/nina/dataset_video_train/metadata.csv")
DEFAULT_OUTPUT = PROJECT_ROOT / "data_splits/source_disjoint_v2"
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=SOURCE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def source_group(sample_id: str) -> str:
    """Group numbered clips from the same named source; keep generic IDs independent."""
    parts = sample_id.split("__")
    if len(parts) >= 3 and re.fullmatch(r"\d+", parts[-1]):
        return "__".join(parts[:-1])
    return sample_id


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_score(
    counts: dict[str, Counter],
    split_name: str,
    group_counts: Counter,
    targets: dict[str, dict[str, float]],
) -> float:
    score = 0.0
    for name in SPLITS:
        for key, target in targets[name].items():
            value = counts[name][key] + (group_counts[key] if name == split_name else 0)
            scale = max(target, 1.0)
            delta = (value - target) / scale
            score += delta * delta
            if value > target:
                score += 1.5 * delta * delta
    return score


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.metadata.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError(f"No metadata rows in {args.metadata}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[source_group(row["sample_id"])].append(row)

    total_counts = Counter({"total": len(rows)})
    for row in rows:
        total_counts[f"bucket:{row['bucket']}"] += 1
        total_counts[f"fallback:{row['reference_is_fallback']}"] += 1
    targets = {
        split: {key: value * RATIOS[split] for key, value in total_counts.items()}
        for split in SPLITS
    }

    rng = random.Random(args.seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    counts = {split: Counter() for split in SPLITS}
    assigned_groups: dict[str, list[str]] = {split: [] for split in SPLITS}
    split_rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for group_name, group_rows in group_items:
        group_counts = Counter({"total": len(group_rows)})
        for row in group_rows:
            group_counts[f"bucket:{row['bucket']}"] += 1
            group_counts[f"fallback:{row['reference_is_fallback']}"] += 1
        candidate_scores = [
            (split_score(counts, split, group_counts, targets), rng.random(), split)
            for split in SPLITS
        ]
        _, _, chosen = min(candidate_scores)
        counts[chosen].update(group_counts)
        assigned_groups[chosen].append(group_name)
        split_rows[chosen].extend(group_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rich_fields = list(rows[0].keys())
    minimal_fields = ["video", "prompt", "reference_image"]
    manifest = {
        "source_metadata": str(args.metadata),
        "seed": args.seed,
        "ratios": RATIOS,
        "group_rule": "IDs with >=3 __-separated parts and a numeric suffix are grouped without the suffix; two-part generic IDs stay independent.",
        "splits": {},
    }
    group_sets = {split: set(assigned_groups[split]) for split in SPLITS}
    if any(group_sets[a] & group_sets[b] for a in SPLITS for b in SPLITS if a < b):
        raise AssertionError("Source-group overlap detected between splits")

    for split in SPLITS:
        ordered = sorted(split_rows[split], key=lambda row: row["sample_id"])
        write_csv(args.output_dir / f"{split}_rich.csv", ordered, rich_fields)
        minimal = [
            {
                "video": row["output_video"],
                "prompt": row["caption"],
                "reference_image": row["reference_image"],
            }
            for row in ordered
        ]
        write_csv(args.output_dir / f"{split}.csv", minimal, minimal_fields)
        bucket_counts = Counter(row["bucket"] for row in ordered)
        fallback_counts = Counter(row["reference_is_fallback"] for row in ordered)
        digest = hashlib.sha256(
            "\n".join(row["sample_id"] for row in ordered).encode("utf-8")
        ).hexdigest()
        manifest["splits"][split] = {
            "rows": len(ordered),
            "ratio": len(ordered) / len(rows),
            "groups": len(assigned_groups[split]),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "reference_is_fallback": dict(sorted(fallback_counts.items())),
            "sample_id_sha256": digest,
        }

    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
