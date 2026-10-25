#!/usr/bin/env python3
"""Rank validated checkpoints with guardrails and a multi-objective Pareto audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DEFAULT_RESULTS = PROJECT_ROOT / "validation_results/formal_val_v2"
OBJECTIVES = (
    "sharpness_retention",
    "motion_retention",
    "warp_error_reduction",
    "target_motion_mae_reduction",
    "reference_ssim_delta",
)
WEIGHTS = {
    "sharpness_retention": 0.20,
    "motion_retention": 0.25,
    "warp_error_reduction": 0.20,
    "target_motion_mae_reduction": 0.25,
    "reference_ssim_delta": 0.10,
}
GUARDRAILS = {
    "minimum_sharpness_retention": 0.85,
    "minimum_motion_retention": 0.70,
    "minimum_reference_ssim_delta": -0.005,
}
BOOTSTRAP_SPECS = {
    "sharpness_retention": ("sharpness_mean", "ratio", 1.0),
    "motion_retention": ("motion_active_fraction", "ratio", 1.0),
    "warp_error_reduction": ("warp_error_mean", "reduction", 0.0),
    "target_motion_mae_reduction": ("target_motion_mean_abs_error", "reduction", 0.0),
    "reference_ssim_delta": ("reference_first_frame_ssim", "delta", 0.0),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def dominates(a: dict, b: dict) -> bool:
    at_least = all(a[key] >= b[key] for key in OBJECTIVES)
    strictly = any(a[key] > b[key] for key in OBJECTIVES)
    return at_least and strictly


def minmax_scores(candidates: list[dict]) -> None:
    for key in OBJECTIVES:
        values = [candidate[key] for candidate in candidates]
        low, high = min(values), max(values)
        for candidate in candidates:
            candidate[f"normalized_{key}"] = (
                1.0 if high == low else (candidate[key] - low) / (high - low)
            )
    for candidate in candidates:
        candidate["exploratory_selection_score"] = sum(
            WEIGHTS[key] * candidate[f"normalized_{key}"] for key in OBJECTIVES
        )


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def transform_metric(base_mean: float, variant_mean: float, transform: str) -> float:
    if transform == "ratio":
        return variant_mean / max(base_mean, 1e-12)
    if transform == "reduction":
        return 1.0 - variant_mean / max(base_mean, 1e-12)
    if transform == "delta":
        return variant_mean - base_mean
    raise ValueError(f"Unknown transform: {transform}")


def paired_bootstrap(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    metric: str,
    transform: str,
    iterations: int,
    seed: int,
    improvement_threshold: float,
) -> dict[str, float]:
    if not pairs:
        raise ValueError("Paired bootstrap requires at least one sample")
    base_values = [f(base, metric) for base, _ in pairs]
    variant_values = [f(variant, metric) for _, variant in pairs]
    count = len(pairs)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(count) for _ in range(count)]
        base_mean = sum(base_values[index] for index in indices) / count
        variant_mean = sum(variant_values[index] for index in indices) / count
        samples.append(transform_metric(base_mean, variant_mean, transform))
    samples.sort()
    point = transform_metric(
        sum(base_values) / count,
        sum(variant_values) / count,
        transform,
    )
    return {
        "point": point,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "probability_improves": sum(value > improvement_threshold for value in samples)
        / iterations,
    }


def paired_rows(rows: list[dict[str, str]], label: str) -> list[tuple[dict[str, str], dict[str, str]]]:
    base_by_id = {row["sample_id"]: row for row in rows if row["variant"] == "base"}
    variant_by_id = {row["sample_id"]: row for row in rows if row["variant"] == label}
    if base_by_id.keys() != variant_by_id.keys():
        missing_base = sorted(variant_by_id.keys() - base_by_id.keys())
        missing_variant = sorted(base_by_id.keys() - variant_by_id.keys())
        raise RuntimeError(
            f"Unpaired rows for {label}: missing_base={missing_base}, missing_variant={missing_variant}"
        )
    return [(base_by_id[sample_id], variant_by_id[sample_id]) for sample_id in sorted(base_by_id)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("--bootstrap-iterations must be at least 1000")

    candidates: list[dict] = []
    for metrics_dir in sorted(args.results_root.glob("metrics_formal_v2_step-*")):
        aggregate_path = metrics_dir / "aggregate_by_variant.csv"
        per_video_path = metrics_dir / "objective_metrics.csv"
        if not aggregate_path.is_file() or not per_video_path.is_file():
            continue
        rows = read_csv(aggregate_path)
        base = next(row for row in rows if row["variant"] == "base")
        variants = [row for row in rows if row["variant"] != "base"]
        if len(variants) != 1:
            raise RuntimeError(f"Expected one checkpoint variant in {metrics_dir}, got {len(variants)}")
        variant = variants[0]
        label = variant["variant"]
        per_video_rows = read_csv(per_video_path)
        pairs = paired_rows(per_video_rows, label)
        label_seed = int.from_bytes(
            hashlib.sha256(label.encode("utf-8")).digest()[:8], "big"
        )
        bootstrap: dict[str, dict[str, float]] = {}
        for index, (objective, (metric, transform, threshold)) in enumerate(
            BOOTSTRAP_SPECS.items()
        ):
            bootstrap[objective] = paired_bootstrap(
                pairs,
                metric,
                transform,
                args.bootstrap_iterations,
                args.bootstrap_seed ^ label_seed ^ index,
                threshold,
            )
        manifest_path = args.results_root / f"validation_manifest_{label}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        sharpness_retention = f(variant, "mean_sharpness_mean") / f(base, "mean_sharpness_mean")
        motion_retention = f(variant, "mean_motion_active_fraction") / max(
            f(base, "mean_motion_active_fraction"), 1e-12
        )
        warp_error_reduction = 1.0 - f(variant, "mean_warp_error_mean") / max(
            f(base, "mean_warp_error_mean"), 1e-12
        )
        target_motion_mae_reduction = 1.0 - f(
            variant, "mean_target_motion_mean_abs_error"
        ) / max(f(base, "mean_target_motion_mean_abs_error"), 1e-12)
        reference_ssim_delta = f(variant, "mean_reference_first_frame_ssim") - f(
            base, "mean_reference_first_frame_ssim"
        )
        eligible = (
            sharpness_retention >= GUARDRAILS["minimum_sharpness_retention"]
            and motion_retention >= GUARDRAILS["minimum_motion_retention"]
            and reference_ssim_delta >= GUARDRAILS["minimum_reference_ssim_delta"]
        )
        eligible_ci = (
            bootstrap["sharpness_retention"]["ci_low"]
            >= GUARDRAILS["minimum_sharpness_retention"]
            and bootstrap["motion_retention"]["ci_low"]
            >= GUARDRAILS["minimum_motion_retention"]
            and bootstrap["reference_ssim_delta"]["ci_low"]
            >= GUARDRAILS["minimum_reference_ssim_delta"]
        )
        candidates.append(
            {
                "label": label,
                "checkpoint": manifest["lora"],
                "samples": int(float(variant["samples"])),
                "eligible_guardrails": eligible,
                "eligible_guardrails_ci": eligible_ci,
                "sharpness_retention": sharpness_retention,
                "motion_retention": motion_retention,
                "warp_error_reduction": warp_error_reduction,
                "target_motion_mae_reduction": target_motion_mae_reduction,
                "reference_ssim_delta": reference_ssim_delta,
                "target_motion_correlation": f(
                    variant, "mean_target_motion_curve_correlation"
                ),
                **{
                    f"{objective}_{suffix}": values[key]
                    for objective, values in bootstrap.items()
                    for suffix, key in (
                        ("ci_low", "ci_low"),
                        ("ci_high", "ci_high"),
                        ("probability_improves", "probability_improves"),
                    )
                },
            }
        )

    if not candidates:
        raise RuntimeError(f"No completed checkpoint metrics found in {args.results_root}")
    for candidate in candidates:
        candidate["pareto_front"] = not any(
            dominates(other, candidate) for other in candidates if other is not candidate
        )
    minmax_scores(candidates)
    candidates.sort(
        key=lambda row: (
            row["eligible_guardrails_ci"],
            row["eligible_guardrails"],
            row["pareto_front"],
            row["exploratory_selection_score"],
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    payload = {
        "guardrails": GUARDRAILS,
        "score_weights": WEIGHTS,
        "score_warning": "The score is exploratory. Final selection requires Pareto review and blinded human evaluation.",
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
            "confidence_interval": 0.95,
            "method": "paired nonparametric bootstrap over held-out samples",
        },
        "ranking": candidates,
    }
    (args.results_root / "checkpoint_ranking.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.results_root / "checkpoint_ranking.csv", candidates)

    lines = [
        "# Formal v2 checkpoint ranking",
        "",
        "The numeric score is exploratory; guardrails, Pareto status, and blind review take precedence.",
        "",
        "| Rank | Checkpoint | Guardrails | CI robust | Pareto | Sharpness (95% CI) | Motion (95% CI) | Warp reduction (95% CI) | Target-motion MAE reduction (95% CI) | Score |",
        "|---:|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['rank']} | `{Path(row['checkpoint']).name}` | "
            f"{'yes' if row['eligible_guardrails'] else 'no'} | "
            f"{'yes' if row['eligible_guardrails_ci'] else 'no'} | "
            f"{'yes' if row['pareto_front'] else 'no'} | "
            f"{row['sharpness_retention']:.3f} [{row['sharpness_retention_ci_low']:.3f}, {row['sharpness_retention_ci_high']:.3f}] | "
            f"{row['motion_retention']:.3f} [{row['motion_retention_ci_low']:.3f}, {row['motion_retention_ci_high']:.3f}] | "
            f"{row['warp_error_reduction']:.3f} [{row['warp_error_reduction_ci_low']:.3f}, {row['warp_error_reduction_ci_high']:.3f}] | "
            f"{row['target_motion_mae_reduction']:.3f} [{row['target_motion_mae_reduction_ci_low']:.3f}, {row['target_motion_mae_reduction_ci_high']:.3f}] | "
            f"{row['exploratory_selection_score']:.3f} |"
        )
    (args.results_root / "checkpoint_ranking.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
