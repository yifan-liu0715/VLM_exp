#!/usr/bin/env python3
"""Aggregate top-checkpoint validation across seeds with hierarchical bootstrap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DEFAULT_FORMAL = PROJECT_ROOT / "validation_results/formal_val_v2"
DEFAULT_MULTI = PROJECT_ROOT / "validation_results/formal_multiseed"
SPECS = {
    "sharpness_retention": ("sharpness_mean", "ratio", 1.0),
    "motion_retention": ("motion_active_fraction", "ratio", 1.0),
    "warp_error_reduction": ("warp_error_mean", "reduction", 0.0),
    "target_motion_mae_reduction": ("target_motion_mean_abs_error", "reduction", 0.0),
    "reference_ssim_delta": ("reference_first_frame_ssim", "delta", 0.0),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def transform(base_mean: float, variant_mean: float, kind: str) -> float:
    if kind == "ratio":
        return variant_mean / max(base_mean, 1e-12)
    if kind == "reduction":
        return 1.0 - variant_mean / max(base_mean, 1e-12)
    if kind == "delta":
        return variant_mean - base_mean
    raise ValueError(kind)


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def pair_rows(rows: list[dict[str, str]], label: str) -> list[tuple[dict, dict]]:
    base = {row["sample_id"]: row for row in rows if row["variant"] == "base"}
    variant = {row["sample_id"]: row for row in rows if row["variant"] == label}
    if base.keys() != variant.keys():
        raise RuntimeError(f"Unpaired metrics for {label}")
    return [(base[sample_id], variant[sample_id]) for sample_id in sorted(base)]


def objective(pairs: list[tuple[dict, dict]], metric: str, kind: str) -> float:
    base_mean = sum(float(base[metric]) for base, _ in pairs) / len(pairs)
    variant_mean = sum(float(variant[metric]) for _, variant in pairs) / len(pairs)
    return transform(base_mean, variant_mean, kind)


def hierarchical_bootstrap(
    pairs_by_seed: dict[int, list[tuple[dict, dict]]],
    metric: str,
    kind: str,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    seed_ids = sorted(pairs_by_seed)
    draws: list[float] = []
    for _ in range(iterations):
        pooled: list[tuple[dict, dict]] = []
        for _ in seed_ids:
            selected_seed = rng.choice(seed_ids)
            pairs = pairs_by_seed[selected_seed]
            pooled.extend(rng.choice(pairs) for _ in pairs)
        draws.append(objective(pooled, metric, kind))
    all_pairs = [pair for pairs in pairs_by_seed.values() for pair in pairs]
    return {
        "point": objective(all_pairs, metric, kind),
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
        "probability_improves": sum(value > threshold for value in draws) / iterations,
    }


def candidate_difference_bootstrap(
    left_by_seed: dict[int, list[tuple[dict, dict]]],
    right_by_seed: dict[int, list[tuple[dict, dict]]],
    metric: str,
    kind: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    seed_ids = sorted(set(left_by_seed) & set(right_by_seed))
    if len(seed_ids) < 2:
        raise RuntimeError("Head-to-head comparison needs at least two shared seeds")
    for seed_id in seed_ids:
        left_ids = [pair[0]["sample_id"] for pair in left_by_seed[seed_id]]
        right_ids = [pair[0]["sample_id"] for pair in right_by_seed[seed_id]]
        if left_ids != right_ids:
            raise RuntimeError(f"Candidate sample mismatch for seed {seed_id}")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        left_pool: list[tuple[dict, dict]] = []
        right_pool: list[tuple[dict, dict]] = []
        for _ in seed_ids:
            seed_id = rng.choice(seed_ids)
            count = len(left_by_seed[seed_id])
            indices = [rng.randrange(count) for _ in range(count)]
            left_pool.extend(left_by_seed[seed_id][index] for index in indices)
            right_pool.extend(right_by_seed[seed_id][index] for index in indices)
        draws.append(
            objective(left_pool, metric, kind) - objective(right_pool, metric, kind)
        )
    all_left = [pair for seed_id in seed_ids for pair in left_by_seed[seed_id]]
    all_right = [pair for seed_id in seed_ids for pair in right_by_seed[seed_id]]
    point = objective(all_left, metric, kind) - objective(all_right, metric, kind)
    draws.sort()
    return {
        "left_minus_right": point,
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
        "probability_left_better": sum(value > 0 for value in draws) / iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--multiseed-root", type=Path, default=DEFAULT_MULTI)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    if args.iterations < 1000:
        raise ValueError("--iterations must be at least 1000")

    ranking = json.loads(
        (args.formal_root / "checkpoint_ranking.json").read_text(encoding="utf-8")
    )["ranking"]
    variants = args.variants or [row["label"] for row in ranking[: args.top_k]]
    roots = [args.formal_root] + sorted(args.multiseed_root.glob("seed-*"))
    rows_out: list[dict] = []
    payload: dict = {"variants": variants, "seeds": [], "results": {}}
    candidate_pairs: dict[str, dict[int, list[tuple[dict, dict]]]] = {}

    for label in variants:
        pairs_by_seed: dict[int, list[tuple[dict, dict]]] = {}
        per_seed: list[dict] = []
        for root in roots:
            manifest_path = root / f"validation_manifest_{label}.json"
            metrics_path = root / f"metrics_{label}" / "objective_metrics.csv"
            if not manifest_path.is_file() or not metrics_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            generation_seed = int(manifest["parameters"]["seed"])
            pairs = pair_rows(read_csv(metrics_path), label)
            pairs_by_seed[generation_seed] = pairs
            seed_row = {"variant": label, "seed": generation_seed, "samples": len(pairs)}
            for name, (metric, kind, _) in SPECS.items():
                seed_row[name] = objective(pairs, metric, kind)
            per_seed.append(seed_row)
            rows_out.append(seed_row)
        if len(pairs_by_seed) < 2:
            raise RuntimeError(f"Need at least two completed seeds for {label}")
        candidate_pairs[label] = pairs_by_seed

        result = {"per_seed": sorted(per_seed, key=lambda row: row["seed"]), "aggregate": {}}
        label_hash = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
        for index, (name, (metric, kind, threshold)) in enumerate(SPECS.items()):
            result["aggregate"][name] = hierarchical_bootstrap(
                pairs_by_seed,
                metric,
                kind,
                threshold,
                args.iterations,
                args.seed ^ label_hash ^ index,
            )
        payload["results"][label] = result
        payload["seeds"] = sorted(set(payload["seeds"]) | set(pairs_by_seed))

    if len(variants) >= 2:
        left, right = variants[:2]
        payload["head_to_head"] = {
            "left": left,
            "right": right,
            "interpretation": "Positive differences and probabilities above 0.5 favor the left candidate.",
            "metrics": {},
        }
        for index, (name, (metric, kind, _)) in enumerate(SPECS.items()):
            payload["head_to_head"]["metrics"][name] = candidate_difference_bootstrap(
                candidate_pairs[left],
                candidate_pairs[right],
                metric,
                kind,
                args.iterations,
                args.seed ^ 0xA5A5A5A5 ^ index,
            )

    args.multiseed_root.mkdir(parents=True, exist_ok=True)
    (args.multiseed_root / "multiseed_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.multiseed_root / "multiseed_per_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    lines = [
        "# Multi-seed robustness summary",
        "",
        f"Seeds: {', '.join(map(str, payload['seeds']))}. Confidence intervals use "
        f"{args.iterations:,} hierarchical bootstrap draws over seeds and samples.",
        "",
        "| Variant | Metric | Point | 95% CI | P(improves) |",
        "|---|---|---:|---:|---:|",
    ]
    for label in variants:
        for name, values in payload["results"][label]["aggregate"].items():
            lines.append(
                f"| `{label}` | {name} | {values['point']:.3f} | "
                f"[{values['ci_low']:.3f}, {values['ci_high']:.3f}] | "
                f"{values['probability_improves']:.3f} |"
            )
    if "head_to_head" in payload:
        comparison = payload["head_to_head"]
        lines.extend(
            [
                "",
                f"## Direct comparison: {comparison['left']} minus {comparison['right']}",
                "",
                "| Metric | Difference | 95% CI | P(left better) |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in comparison["metrics"].items():
            lines.append(
                f"| {name} | {values['left_minus_right']:+.3f} | "
                f"[{values['ci_low']:+.3f}, {values['ci_high']:+.3f}] | "
                f"{values['probability_left_better']:.3f} |"
            )
    (args.multiseed_root / "multiseed_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
