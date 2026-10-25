#!/usr/bin/env python3
"""Direct paired-bootstrap comparison of the top two candidates in one validation run."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


SPECS = {
    "sharpness_retention": ("sharpness_mean", "ratio"),
    "motion_retention": ("motion_active_fraction", "ratio"),
    "warp_error_reduction": ("warp_error_mean", "reduction"),
    "target_motion_mae_reduction": ("target_motion_mean_abs_error", "reduction"),
    "reference_ssim_delta": ("reference_first_frame_ssim", "delta"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def transform(base: float, variant: float, kind: str) -> float:
    if kind == "ratio":
        return variant / max(base, 1e-12)
    if kind == "reduction":
        return 1.0 - variant / max(base, 1e-12)
    if kind == "delta":
        return variant - base
    raise ValueError(kind)


def objective(pairs: list[tuple[dict, dict]], metric: str, kind: str) -> float:
    base = sum(float(left[metric]) for left, _ in pairs) / len(pairs)
    variant = sum(float(right[metric]) for _, right in pairs) / len(pairs)
    return transform(base, variant, kind)


def load_pairs(results: Path, label: str) -> list[tuple[dict, dict]]:
    rows = read_csv(results / f"metrics_{label}/objective_metrics.csv")
    base = {row["sample_id"]: row for row in rows if row["variant"] == "base"}
    variant = {row["sample_id"]: row for row in rows if row["variant"] == label}
    if base.keys() != variant.keys():
        raise RuntimeError(f"Unpaired metrics for {label}")
    return [(base[sample_id], variant[sample_id]) for sample_id in sorted(base)]


def percentile(values: list[float], probability: float) -> float:
    values.sort()
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--left", default=None)
    parser.add_argument("--right", default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    ranking = json.loads(
        (args.results_root / "checkpoint_ranking.json").read_text(encoding="utf-8")
    )["ranking"]
    left = args.left or ranking[0]["label"]
    right = args.right or ranking[1]["label"]
    left_pairs = load_pairs(args.results_root, left)
    right_pairs = load_pairs(args.results_root, right)
    left_ids = [pair[0]["sample_id"] for pair in left_pairs]
    right_ids = [pair[0]["sample_id"] for pair in right_pairs]
    if left_ids != right_ids:
        raise RuntimeError("Candidate samples do not match")

    rng = random.Random(args.seed)
    results = {}
    count = len(left_pairs)
    for name, (metric, kind) in SPECS.items():
        draws = []
        for _ in range(args.iterations):
            indices = [rng.randrange(count) for _ in range(count)]
            left_draw = [left_pairs[index] for index in indices]
            right_draw = [right_pairs[index] for index in indices]
            draws.append(
                objective(left_draw, metric, kind) - objective(right_draw, metric, kind)
            )
        point = objective(left_pairs, metric, kind) - objective(right_pairs, metric, kind)
        results[name] = {
            "left_minus_right": point,
            "ci_low": percentile(draws, 0.025),
            "ci_high": percentile(draws, 0.975),
            "probability_left_better": sum(value > 0 for value in draws) / len(draws),
        }

    payload = {
        "left": left,
        "right": right,
        "samples": count,
        "bootstrap_iterations": args.iterations,
        "metrics": results,
    }
    (args.results_root / "candidate_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Candidate comparison: {left} minus {right}",
        "",
        "Positive differences favor the left candidate.",
        "",
        "| Metric | Difference | 95% CI | P(left better) |",
        "|---|---:|---:|---:|",
    ]
    for name, values in results.items():
        lines.append(
            f"| {name} | {values['left_minus_right']:+.3f} | "
            f"[{values['ci_low']:+.3f}, {values['ci_high']:+.3f}] | "
            f"{values['probability_left_better']:.3f} |"
        )
    (args.results_root / "candidate_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
