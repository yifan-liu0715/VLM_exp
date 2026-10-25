#!/usr/bin/env python3
"""Audit checkpoint effects by resolution bucket and semantic content group."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DEFAULT_RESULTS = PROJECT_ROOT / "validation_results/formal_val_v2"
DEFAULT_SUBSET = PROJECT_ROOT / "data_splits/source_disjoint_v2/validation_subset_15.json"
SPECS = {
    "sharpness_retention": ("sharpness_mean", "ratio"),
    "motion_retention": ("motion_active_fraction", "ratio"),
    "warp_error_reduction": ("warp_error_mean", "reduction"),
    "target_motion_mae_reduction": ("target_motion_mean_abs_error", "reduction"),
    "reference_ssim_delta": ("reference_first_frame_ssim", "delta"),
}
GUARDRAILS = {
    "sharpness_retention": 0.85,
    "motion_retention": 0.70,
    "reference_ssim_delta": -0.005,
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


def aggregate(pairs: list[tuple[dict, dict]]) -> dict[str, float]:
    result = {}
    for name, (metric, kind) in SPECS.items():
        base = sum(float(left[metric]) for left, _ in pairs) / len(pairs)
        variant = sum(float(right[metric]) for _, right in pairs) / len(pairs)
        result[name] = transform(base, variant, kind)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    args = parser.parse_args()

    samples = json.loads(args.subset.read_text(encoding="utf-8"))["samples"]
    metadata = {
        sample["sample_id"]: {
            "bucket": sample["bucket"],
            "content": "dance" if "__humandance__" in sample["sample_id"] else "talking_face",
        }
        for sample in samples
    }
    output_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for metrics_dir in sorted(args.results_root.glob("metrics_formal_v2_step-*")):
        path = metrics_dir / "objective_metrics.csv"
        if not path.is_file():
            continue
        rows = read_csv(path)
        labels = sorted({row["variant"] for row in rows if row["variant"] != "base"})
        if len(labels) != 1:
            raise RuntimeError(f"Expected one non-base variant in {path}")
        label = labels[0]
        base = {row["sample_id"]: row for row in rows if row["variant"] == "base"}
        variant = {row["sample_id"]: row for row in rows if row["variant"] == label}
        variant_rows: list[dict] = []
        for group_type in ("bucket", "content"):
            group_values = sorted({metadata[sample_id][group_type] for sample_id in base})
            for group_value in group_values:
                sample_ids = [
                    sample_id
                    for sample_id in base
                    if metadata[sample_id][group_type] == group_value
                ]
                pairs = [(base[sample_id], variant[sample_id]) for sample_id in sample_ids]
                metrics = aggregate(pairs)
                row = {
                    "variant": label,
                    "group_type": group_type,
                    "group": group_value,
                    "samples": len(pairs),
                    **metrics,
                    "passes_group_guardrails": all(
                        metrics[name] >= threshold for name, threshold in GUARDRAILS.items()
                    ),
                }
                output_rows.append(row)
                variant_rows.append(row)
        summary[label] = {
            "all_subgroups_pass": all(row["passes_group_guardrails"] for row in variant_rows),
            "worst_sharpness_retention": min(row["sharpness_retention"] for row in variant_rows),
            "worst_motion_retention": min(row["motion_retention"] for row in variant_rows),
            "worst_reference_ssim_delta": min(row["reference_ssim_delta"] for row in variant_rows),
            "failed_subgroups": [
                f"{row['group_type']}={row['group']}"
                for row in variant_rows
                if not row["passes_group_guardrails"]
            ],
        }

    if not output_rows:
        raise RuntimeError("No complete checkpoint metrics found")
    with (args.results_root / "subgroup_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    payload = {"guardrails": GUARDRAILS, "summary": summary, "rows": output_rows}
    (args.results_root / "subgroup_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Formal validation subgroup audit",
        "",
        "A checkpoint should not be selected solely from the global mean. Subgroup failures are veto signals.",
        "",
        "| Variant | Group | N | Sharpness | Motion | Warp reduction | Target-motion MAE reduction | Ref SSIM Δ | Guardrails |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in output_rows:
        lines.append(
            f"| `{row['variant']}` | {row['group_type']}={row['group']} | {row['samples']} | "
            f"{row['sharpness_retention']:.3f} | {row['motion_retention']:.3f} | "
            f"{row['warp_error_reduction']:.3f} | {row['target_motion_mae_reduction']:.3f} | "
            f"{row['reference_ssim_delta']:+.4f} | "
            f"{'pass' if row['passes_group_guardrails'] else 'fail'} |"
        )
    (args.results_root / "subgroup_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
