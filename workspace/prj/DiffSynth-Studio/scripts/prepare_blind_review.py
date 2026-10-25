#!/usr/bin/env python3
"""Create deterministic, label-hidden A/B videos and review sheets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DEFAULT_RESULTS = PROJECT_ROOT / "validation_results/formal_val_v2"
DEFAULT_SUBSET = PROJECT_ROOT / "data_splits/source_disjoint_v2/validation_subset_15.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Explicit variant labels. Default: top eligible Pareto variants from ranking.",
    )
    parser.add_argument("--max-variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Only write manifests/sheets; do not invoke ffmpeg.",
    )
    return parser.parse_args()


def choose_variants(results_root: Path, explicit: list[str] | None, maximum: int) -> list[str]:
    if maximum < 1:
        raise ValueError("--max-variants must be at least 1")
    if explicit:
        return explicit[:maximum]

    ranking_path = results_root / "checkpoint_ranking.json"
    if not ranking_path.is_file():
        raise FileNotFoundError(
            f"Ranking does not exist: {ranking_path}. Pass --variants explicitly or run ranking first."
        )
    rows = json.loads(ranking_path.read_text(encoding="utf-8"))["ranking"]
    tiers = (
        [row for row in rows if row["eligible_guardrails"] and row["pareto_front"]],
        [row for row in rows if row["eligible_guardrails"]],
        rows,
    )
    selected: list[str] = []
    for tier in tiers:
        for row in tier:
            label = row["label"]
            if label not in selected:
                selected.append(label)
            if len(selected) == maximum:
                return selected
    return selected


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_pair(left: Path, right: Path, output: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS,scale=576:320[a];"
        "[1:v]setpts=PTS-STARTPTS,scale=576:320[b];"
        "[a][b]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-r",
        "15",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_root / "blind_review"
    )
    if not args.no_render and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required unless --no-render is set")

    subset = json.loads(args.subset.read_text(encoding="utf-8"))["samples"]
    variants = choose_variants(results_root, args.variants, args.max_variants)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "pairs"
    video_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    assignments: dict[tuple[str, str], bool] = {}
    for variant in variants:
        samples = list(subset)
        rng.shuffle(samples)
        # Approximately equal left/right placement per checkpoint avoids side bias.
        for index, sample in enumerate(samples):
            assignments[(variant, sample["sample_id"])] = index % 2 == 0

    pairs: list[dict] = []
    key_rows: list[dict] = []
    review_rows: list[dict] = []
    pair_number = 0
    for variant in variants:
        for sample in subset:
            sample_id = sample["sample_id"]
            base_path = results_root / f"{sample_id}_base.mp4"
            variant_path = results_root / f"{sample_id}_{variant}.mp4"
            missing = [str(path) for path in (base_path, variant_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing validation video(s):\n" + "\n".join(missing))

            pair_number += 1
            pair_id = f"pair_{pair_number:04d}"
            variant_on_left = assignments[(variant, sample_id)]
            left_path, right_path = (
                (variant_path, base_path) if variant_on_left else (base_path, variant_path)
            )
            left_variant, right_variant = (
                (variant, "base") if variant_on_left else ("base", variant)
            )
            pair_path = video_dir / f"{pair_id}.mp4"
            if not args.no_render and (args.overwrite or not pair_path.exists()):
                render_pair(left_path, right_path, pair_path)

            pairs.append(
                {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "bucket": sample["bucket"],
                    "prompt": sample["caption"],
                    "video": str(pair_path),
                }
            )
            key_rows.append(
                {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                }
            )
            review_rows.append(
                {
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "bucket": sample["bucket"],
                    "prompt": sample["caption"],
                    "pair_video": str(pair_path),
                    "identity_preference_left_right_tie": "",
                    "motion_alignment_preference_left_right_tie": "",
                    "temporal_coherence_preference_left_right_tie": "",
                    "visual_quality_preference_left_right_tie": "",
                    "overall_preference_left_right_tie": "",
                    "notes": "",
                }
            )

    manifest = {
        "seed": args.seed,
        "left_is_A_right_is_B": True,
        "variants": variants,
        "samples": len(subset),
        "pairs": pairs,
        "instructions": [
            "Reviewers receive blind_review_sheet.csv and pairs/, but not blind_review_key.csv.",
            "For every preference column, enter left, right, or tie.",
            "Judge identity, prompt-aligned motion, temporal coherence, visual quality, and overall preference independently.",
            "Shuffle pair viewing order per reviewer; do not reveal checkpoint labels.",
        ],
    }
    (output_dir / "blind_review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        output_dir / "blind_review_key.csv",
        key_rows,
        ["pair_id", "sample_id", "left_variant", "right_variant"],
    )
    write_csv(
        output_dir / "blind_review_sheet.csv",
        review_rows,
        list(review_rows[0].keys()),
    )
    print(
        f"Prepared {len(pairs)} blind pairs for {len(variants)} variants in {output_dir}"
    )


if __name__ == "__main__":
    main()
