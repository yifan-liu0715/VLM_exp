#!/usr/bin/env python3
"""Analyze one or more completed hidden-label A/B review sheets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


DIMENSIONS = (
    "identity_preference_left_right_tie",
    "motion_alignment_preference_left_right_tie",
    "temporal_coherence_preference_left_right_tie",
    "visual_quality_preference_left_right_tie",
    "overall_preference_left_right_tie",
)
VALID = {"left", "right", "tie"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def exact_sign_test(wins: int, losses: int) -> float:
    count = wins + losses
    if count == 0:
        return 1.0
    lower = min(wins, losses)
    tail = sum(math.comb(count, value) for value in range(lower + 1)) / (2**count)
    return min(1.0, 2.0 * tail)


def score(responses: list[dict]) -> float:
    return sum(1.0 if row["outcome"] == "win" else 0.5 if row["outcome"] == "tie" else 0.0 for row in responses) / len(responses)


def cluster_bootstrap(
    responses: list[dict], iterations: int, seed: int
) -> tuple[float, float]:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for response in responses:
        by_sample[response["sample_id"]].append(response)
    sample_ids = sorted(by_sample)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [rng.choice(sample_ids) for _ in sample_ids]
        draw = [response for sample_id in sampled for response in by_sample[sample_id]]
        draws.append(score(draw))
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("--bootstrap-iterations must be at least 1000")

    key = {row["pair_id"]: row for row in read_csv(args.key)}
    responses: list[dict] = []
    reviewer_choices: dict[tuple[str, str], list[str]] = defaultdict(list)
    for reviewer_index, sheet in enumerate(args.sheets, 1):
        for row in read_csv(sheet):
            pair_id = row["pair_id"]
            if pair_id not in key:
                raise ValueError(f"Unknown pair_id {pair_id!r} in {sheet}")
            mapping = key[pair_id]
            variants = {mapping["left_variant"], mapping["right_variant"]} - {"base"}
            if len(variants) != 1:
                raise ValueError(f"Pair {pair_id} is not base-vs-one-variant")
            variant = variants.pop()
            for dimension in DIMENSIONS:
                choice = row.get(dimension, "").strip().lower()
                if choice not in VALID:
                    raise ValueError(
                        f"{sheet}: pair {pair_id}, {dimension} must be left/right/tie, got {choice!r}"
                    )
                winner = mapping[f"{choice}_variant"] if choice != "tie" else "tie"
                outcome = "tie" if winner == "tie" else "win" if winner == variant else "loss"
                responses.append(
                    {
                        "reviewer": reviewer_index,
                        "pair_id": pair_id,
                        "sample_id": row["sample_id"],
                        "variant": variant,
                        "dimension": dimension.removesuffix("_preference_left_right_tie"),
                        "outcome": outcome,
                    }
                )
                reviewer_choices[(pair_id, dimension)].append(choice)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for response in responses:
        grouped[(response["variant"], response["dimension"])].append(response)
    summaries = []
    for index, ((variant, dimension), rows) in enumerate(sorted(grouped.items())):
        wins = sum(row["outcome"] == "win" for row in rows)
        losses = sum(row["outcome"] == "loss" for row in rows)
        ties = sum(row["outcome"] == "tie" for row in rows)
        low, high = cluster_bootstrap(
            rows, args.bootstrap_iterations, args.seed ^ index
        )
        summaries.append(
            {
                "variant": variant,
                "dimension": dimension,
                "responses": len(rows),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "tie_split_win_rate": score(rows),
                "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
                "exact_sign_test_p": exact_sign_test(wins, losses),
            }
        )

    agreements = []
    for (pair_id, dimension), choices in reviewer_choices.items():
        if len(choices) < 2:
            continue
        pairs = [(choices[i], choices[j]) for i in range(len(choices)) for j in range(i + 1, len(choices))]
        agreements.append(sum(left == right for left, right in pairs) / len(pairs))
    payload = {
        "reviewers": len(args.sheets),
        "bootstrap_iterations": args.bootstrap_iterations,
        "mean_pairwise_reviewer_agreement": (
            sum(agreements) / len(agreements) if agreements else None
        ),
        "summary": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "blind_review_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "blind_review_analysis.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Blind review analysis",
        "",
        f"Reviewers: {len(args.sheets)}. Mean pairwise agreement: "
        f"{payload['mean_pairwise_reviewer_agreement'] if agreements else 'not available'}.",
        "",
        "| Variant | Dimension | W/L/T | Tie-split win rate | 95% cluster CI | Sign-test p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['variant']}` | {row['dimension']} | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | "
            f"{row['tie_split_win_rate']:.3f} | "
            f"[{row['cluster_bootstrap_ci_low']:.3f}, {row['cluster_bootstrap_ci_high']:.3f}] | "
            f"{row['exact_sign_test_p']:.4f} |"
        )
    (args.output_dir / "blind_review_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
