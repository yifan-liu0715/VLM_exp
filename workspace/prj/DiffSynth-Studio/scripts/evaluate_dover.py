#!/usr/bin/env python3
"""Batch-score formal base/LoRA videos with the official DOVER-Mobile model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DEFAULT_RESULTS = PROJECT_ROOT / "validation_results/formal_val_v2"
DEFAULT_DOVER = PROJECT_ROOT / "third_party/DOVER"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--ranking", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dover-root", type=Path, default=DEFAULT_DOVER)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fused_score(technical: float, aesthetic: float) -> float:
    value = (technical - 0.1107) / 0.07355 * 0.6104 + (
        aesthetic + 0.08285
    ) / 0.03774 * 0.3896
    return 1.0 / (1.0 + math.exp(-value))


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    dover_root = args.dover_root.resolve()
    ranking_path = args.ranking or results_root / "checkpoint_ranking.json"
    config_path = args.config or dover_root / "dover-mobile.yml"
    weights_path = args.weights or dover_root / "pretrained_weights/DOVER-Mobile.pth"
    output_dir = args.output_dir or results_root / "dover_mobile"
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (ranking_path, config_path, weights_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))["ranking"]
    variants = [row["label"] for row in ranking[: args.top_k]]
    items: dict[str, dict] = {}
    for variant in variants:
        manifest = json.loads(
            (results_root / f"validation_manifest_{variant}.json").read_text(
                encoding="utf-8"
            )
        )
        for item in manifest["outputs"]:
            if item["variant"] not in ("base", variant):
                continue
            path = str(Path(item["path"]).resolve())
            items[path] = {
                "sample_id": item["sample_id"],
                "variant": item["variant"],
                "path": path,
            }

    sys.path.insert(0, str(dover_root))
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import DOVER

    options = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluator = DOVER(**options["model"]["args"]).to(args.device)
    evaluator.load_state_dict(torch.load(weights_path, map_location=args.device))
    evaluator.eval()
    data_options = options["data"]["val-l1080p"]["args"]
    temporal_samplers = {}
    for sample_type, sample_options in data_options["sample_types"].items():
        if "t_frag" not in sample_options:
            temporal_samplers[sample_type] = UnifiedFrameSampler(
                sample_options["clip_len"],
                sample_options["num_clips"],
                sample_options["frame_interval"],
            )
        else:
            temporal_samplers[sample_type] = UnifiedFrameSampler(
                sample_options["clip_len"] // sample_options["t_frag"],
                sample_options["t_frag"],
                sample_options["frame_interval"],
                sample_options["num_clips"],
            )

    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])
    rows: list[dict] = []
    with torch.inference_mode():
        for index, item in enumerate(sorted(items.values(), key=lambda row: (row["sample_id"], row["variant"])), 1):
            views, _ = spatial_temporal_view_decomposition(
                item["path"], data_options["sample_types"], temporal_samplers
            )
            for key, value in views.items():
                clips = data_options["sample_types"][key].get("num_clips", 1)
                views[key] = (
                    ((value.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(value.shape[0], clips, -1, *value.shape[2:])
                    .transpose(0, 1)
                    .to(args.device)
                )
            technical, aesthetic = [score.mean().item() for score in evaluator(views)]
            row = {
                **item,
                "dover_technical": technical,
                "dover_aesthetic": aesthetic,
                "dover_fused": fused_score(technical, aesthetic),
            }
            rows.append(row)
            print(
                f"[{index}/{len(items)}] {item['sample_id']} {item['variant']} "
                f"fused={row['dover_fused']:.4f}",
                flush=True,
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    aggregate = []
    for variant, variant_rows in sorted(grouped.items()):
        aggregate.append(
            {
                "variant": variant,
                "samples": len(variant_rows),
                "mean_dover_technical": sum(row["dover_technical"] for row in variant_rows)
                / len(variant_rows),
                "mean_dover_aesthetic": sum(row["dover_aesthetic"] for row in variant_rows)
                / len(variant_rows),
                "mean_dover_fused": sum(row["dover_fused"] for row in variant_rows)
                / len(variant_rows),
            }
        )
    payload = {
        "model": "DOVER-Mobile v0.5.0",
        "official_source": "https://github.com/VQAssessment/DOVER",
        "device": args.device,
        "variants": variants,
        "per_video": rows,
        "aggregate": aggregate,
        "interpretation": "Secondary learned quality signal; blind human review remains decisive.",
    }
    (output_dir / "dover_scores.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "dover_per_video.csv", rows)
    write_csv(output_dir / "dover_aggregate.csv", aggregate)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/build_formal_dashboard.py"),
            "--results-root",
            str(results_root),
        ],
        check=True,
    )
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
