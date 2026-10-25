#!/usr/bin/env python3
"""Reference and temporal metrics for Wan2.2 base-vs-LoRA validation videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DATASET_ROOT = Path("/nvmedata/workspace2/users/nina/dataset_video_train")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RESULTS_ROOT = PROJECT_ROOT / "validation_results"
MANIFEST_PATH = RESULTS_ROOT / "validation_manifest.json"
METRICS_ROOT = RESULTS_ROOT / "metrics"
WIDTH = 576
HEIGHT = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=METRICS_ROOT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--split-name",
        type=str,
        default="unspecified",
        help="Human-readable evaluation split identifier recorded in the protocol.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for base/target optical flow; defaults inside --results-dir.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["lora"],
        help="LoRA output filename variants to compare against base.",
    )
    return parser.parse_args()


def read_video(
    path: Path, resize_to: tuple[int, int] | None = None
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if resize_to is not None and (frame.shape[1], frame.shape[0]) != resize_to:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"Video has fewer than 2 decodable frames: {path}")
    return frames


def sample_uniform(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[index] for index in indices]


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = to_gray(image_a)
    b = to_gray(image_b)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def motion_and_warp(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    grays = [to_gray(frame) for frame in frames]
    height, width = grays[0].shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    magnitudes: list[float] = []
    warp_errors: list[float] = []
    for previous, current in zip(grays[:-1], grays[1:]):
        forward = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitudes.append(float(np.linalg.norm(forward, axis=2).mean()))

        backward = cv2.calcOpticalFlowFarneback(
            current, previous, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        map_x = (grid_x + backward[..., 0]).astype(np.float32)
        map_y = (grid_y + backward[..., 1]).astype(np.float32)
        warped_previous = cv2.remap(
            previous,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        warp_errors.append(float(np.mean(np.abs(current - warped_previous)) / 255.0))
    return np.asarray(magnitudes), np.asarray(warp_errors)


def video_metrics(
    frames: list[np.ndarray],
    reference: np.ndarray,
    motion: np.ndarray | None = None,
    warp_error: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    if motion is None or warp_error is None:
        motion, warp_error = motion_and_warp(frames)
    sharpness = np.asarray(
        [cv2.Laplacian(to_gray(frame), cv2.CV_32F).var() for frame in frames]
    )
    temporal_abs_diff = np.asarray(
        [
            np.mean(np.abs(to_gray(current) - to_gray(previous))) / 255.0
            for previous, current in zip(frames[:-1], frames[1:])
        ]
    )
    metrics = {
        "reference_first_frame_ssim": ssim(frames[0], reference),
        "sharpness_mean": float(sharpness.mean()),
        "sharpness_std": float(sharpness.std()),
        "flow_mean": float(motion.mean()),
        "flow_p95": float(np.percentile(motion, 95)),
        "motion_active_fraction": float(np.mean(motion > 0.5)),
        "warp_error_mean": float(warp_error.mean()),
        "temporal_abs_diff_mean": float(temporal_abs_diff.mean()),
    }
    return metrics, motion


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size != b.size or a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata_rows = {
        row["sample_id"]: row
        for row in csv.DictReader(args.metadata.open("r", encoding="utf-8", newline=""))
    }
    sample_ids = list(dict.fromkeys(item["sample_id"] for item in manifest["outputs"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else args.results_dir.resolve() / ".evaluation_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    summaries: list[dict] = []
    evaluated_resolutions: set[tuple[int, int]] = set()
    for sample_id in sample_ids:
        metadata = metadata_rows[sample_id]
        reference_path = DATASET_ROOT / metadata["reference_image"]
        target_path = DATASET_ROOT / metadata["output_video"]
        base_frames = read_video(args.results_dir / f"{sample_id}_base.mp4")
        height, width = base_frames[0].shape[:2]
        evaluated_resolutions.add((width, height))
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise RuntimeError(f"Cannot read reference image: {reference_path}")
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)

        cache_path = cache_dir / f"{sample_id}.npz"
        if cache_path.is_file():
            with np.load(cache_path) as cached:
                target_motion = cached["target_motion"]
                base_motion = cached["base_motion"]
                base_warp_error = cached["base_warp_error"]
            expected = len(base_frames) - 1
            if any(array.size != expected for array in (target_motion, base_motion, base_warp_error)):
                raise RuntimeError(f"Stale motion cache for {sample_id}: {cache_path}")
        else:
            target_frames = sample_uniform(
                read_video(target_path, (width, height)), len(base_frames)
            )
            target_motion, _ = motion_and_warp(target_frames)
            base_motion, base_warp_error = motion_and_warp(base_frames)
            temporary = cache_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    target_motion=target_motion,
                    base_motion=base_motion,
                    base_warp_error=base_warp_error,
                )
            temporary.replace(cache_path)
        variant_frames = {"base": base_frames}
        for variant in args.variants:
            variant_frames[variant] = read_video(
                args.results_dir / f"{sample_id}_{variant}.mp4", (width, height)
            )

        sample_metrics: dict[str, dict[str, float | None]] = {}
        for variant, frames in variant_frames.items():
            if variant == "base":
                metrics, motion = video_metrics(
                    frames, reference, base_motion, base_warp_error
                )
            else:
                metrics, motion = video_metrics(frames, reference)
            metrics.update(
                {
                    "target_motion_curve_correlation": pearson_safe(motion, target_motion),
                    "target_motion_mean_abs_error": float(np.mean(np.abs(motion - target_motion))),
                    "target_motion_ratio": float(motion.mean() / max(target_motion.mean(), 1e-8)),
                }
            )
            sample_metrics[variant] = metrics
            metric_rows.append({"sample_id": sample_id, "variant": variant, **metrics})

        for variant in args.variants:
            paired_ssim = float(
                np.mean([ssim(a, b) for a, b in zip(base_frames, variant_frames[variant])])
            )
            summary = {
                "sample_id": sample_id,
                "variant": variant,
                "base_variant_frame_ssim": paired_ssim,
            }
            for metric_name in sample_metrics["base"]:
                base_value = sample_metrics["base"][metric_name]
                variant_value = sample_metrics[variant][metric_name]
                summary[f"base_{metric_name}"] = base_value
                summary[f"variant_{metric_name}"] = variant_value
                summary[f"delta_variant_minus_base_{metric_name}"] = (
                    None
                    if base_value is None or variant_value is None
                    else variant_value - base_value
                )
            summaries.append(summary)

    payload = {
        "protocol": {
            "evaluated_resolutions": [list(size) for size in sorted(evaluated_resolutions)],
            "evaluation_split": args.split_name,
            "target_alignment": "uniformly sample target video to generated frame count",
            "limitations": [
                "No learned perceptual or prompt-alignment metric is available in the current environment.",
                "Optical-flow and sharpness proxies must be interpreted jointly with human review.",
            ],
        },
        "per_video": metric_rows,
        "per_sample_comparison": summaries,
    }
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in metric_rows:
        grouped_rows[row["variant"]].append(row)
    aggregate_rows: list[dict] = []
    for variant, rows in grouped_rows.items():
        aggregate = {"variant": variant, "samples": len(rows)}
        for key in rows[0]:
            if key in {"sample_id", "variant"}:
                continue
            values = [row[key] for row in rows if row[key] is not None]
            aggregate[f"mean_{key}"] = float(np.mean(values)) if values else None
        aggregate_rows.append(aggregate)
    aggregate_rows.sort(key=lambda row: row["variant"])
    payload["aggregate_by_variant"] = aggregate_rows
    (args.output_dir / "objective_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_csv(args.output_dir / "objective_metrics.csv", metric_rows)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "aggregate_by_variant.csv", aggregate_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
