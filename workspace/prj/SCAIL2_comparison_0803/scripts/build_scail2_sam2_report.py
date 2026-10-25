#!/usr/bin/env python3
"""Build a labeled, auditable SCAIL-2 Base/DPO comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam2_fallback_0808.inputs import load_sample
from scripts.run_scail2_sam2_gate import assert_fair_pair, option_map


MASK_LABEL = "official SAM2 fallback; not official SAM3"
CONDITION_OPTIONS = ("--image", "--mask_image", "--pose", "--mask_video")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pair_metadata(pair_manifest: dict, condition_hashes: dict) -> None:
    if pair_manifest.get("mask_preprocessor") != MASK_LABEL:
        raise ValueError("pair manifest is missing the SAM2 fallback label")
    if pair_manifest.get("fairness_check") != "passed":
        raise ValueError("pair manifest fairness check is not passed")
    by_variant = {item.get("variant"): item for item in pair_manifest.get("runs", [])}
    if set(by_variant) != {"base", "dpo"}:
        raise ValueError("pair manifest must contain Base and DPO")
    base_command = by_variant["base"].get("command", [])
    dpo_command = by_variant["dpo"].get("command", [])
    if len(base_command) < 2 or len(dpo_command) < 2:
        raise ValueError("pair manifest commands are incomplete")
    assert_fair_pair(
        {"argv": list(base_command[1:])},
        {"argv": list(dpo_command[1:])},
    )
    if condition_hashes.get("base") != condition_hashes.get("dpo"):
        raise ValueError("Base/DPO condition hashes differ")


def _ffprobe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def _frame_count(probe: dict) -> int:
    stream = probe["streams"][0]
    return int(stream.get("nb_read_frames") or stream.get("nb_frames"))


def _fps(probe: dict) -> float:
    stream = probe["streams"][0]
    return float(Fraction(stream.get("avg_frame_rate") or stream["r_frame_rate"]))


def _condition_hashes(command: list[str]) -> dict[str, str]:
    _, options = option_map(command[1:])
    hashes = {}
    for name in CONDITION_OPTIONS:
        path = Path(str(options[name]))
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256_file(path)
    return hashes


def _run_by_variant(pair_manifest: dict) -> dict[str, dict]:
    return {item["variant"]: item for item in pair_manifest["runs"]}


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"invalid frame shape: {value.shape}")
    scale = min(width / value.shape[1], height / value.shape[0])
    resized_w = max(1, int(round(value.shape[1] * scale)))
    resized_h = max(1, int(round(value.shape[0] * scale)))
    resized = cv2.resize(value, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def _labeled_panel(frame: np.ndarray, label: str, width: int = 512, height: int = 704) -> np.ndarray:
    body = _fit(frame, width, height)
    canvas = np.zeros((height + 40, width, 3), dtype=np.uint8)
    canvas[40:] = body
    cv2.putText(canvas, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    return capture


def _encode_grid_frames(frame_dir: Path, fps: float, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", f"{fps:g}", "-i", str(frame_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output),
        ],
        check=True,
    )


def build_report(root: Path, sample_id: str, output: Path) -> dict:
    sample = load_sample(root, sample_id)
    pair_dir = root / "outputs" / "sam2_fallback_base_vs_dpo" / sample_id
    pair_path = pair_dir / "run_manifest.json"
    mask_path = root / "preprocessed_sam2" / sample_id / "mask_manifest.json"
    pair_manifest = json.loads(pair_path.read_text(encoding="utf-8"))
    mask_manifest = json.loads(mask_path.read_text(encoding="utf-8"))
    if mask_manifest.get("status") != "passed" or mask_manifest.get("mask_preprocessor") != MASK_LABEL:
        raise ValueError("approved SAM2 fallback mask manifest required")

    runs = _run_by_variant(pair_manifest)
    condition_hashes = {
        name: _condition_hashes(runs[name]["command"]) for name in ("base", "dpo")
    }
    validate_pair_metadata(pair_manifest, condition_hashes)

    paths = {
        "source": sample.source_video,
        "target": sample.target_video,
        "base": Path(runs["base"]["output"]),
        "dpo": Path(runs["dpo"]["output"]),
    }
    probes = {name: _ffprobe(path) for name, path in paths.items()}
    for name, probe in probes.items():
        stream = probe["streams"][0]
        if (int(stream["width"]), int(stream["height"])) != (sample.width, sample.height):
            raise ValueError(f"{name} dimensions do not match sample")
        count = _frame_count(probe)
        if count < sample.num_frames or (name in {"base", "dpo"} and count != sample.num_frames):
            raise ValueError(f"{name} frame count is invalid: {count}")
        if abs(_fps(probe) - sample.fps) > 0.01:
            raise ValueError(f"{name} fps does not match sample")

    reference = cv2.imread(str(sample.reference_image), cv2.IMREAD_COLOR)
    if reference is None:
        raise ValueError(f"cannot read reference: {sample.reference_image}")
    captures = {name: _open_video(path) for name, path in paths.items()}
    output.mkdir(parents=True, exist_ok=True)
    frames_output = output / "frames"
    frames_output.mkdir(parents=True, exist_ok=True)
    sampled_stdev = {"base": {}, "dpo": {}}
    comparison = output / "comparison_5grid.mp4"
    labels = {
        "source": "Source (conditioning)",
        "reference": "Reference (conditioning)",
        "target": "Target (evaluation only)",
        "base": "Base (SAM2 fallback masks)",
        "dpo": "DPO (SAM2 fallback masks)",
    }
    try:
        with tempfile.TemporaryDirectory(dir=output, prefix="grid_frames_") as tmp:
            frame_dir = Path(tmp)
            for index in range(sample.num_frames):
                decoded = {}
                for name, capture in captures.items():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise ValueError(f"cannot decode {name} frame {index}")
                    decoded[name] = frame
                panels = [
                    _labeled_panel(decoded["source"], labels["source"]),
                    _labeled_panel(reference, labels["reference"]),
                    _labeled_panel(decoded["target"], labels["target"]),
                    _labeled_panel(decoded["base"], labels["base"]),
                    _labeled_panel(decoded["dpo"], labels["dpo"]),
                ]
                grid = np.hstack(panels)
                frame_path = frame_dir / f"frame_{index:05d}.png"
                if not cv2.imwrite(str(frame_path), grid):
                    raise RuntimeError(f"failed to write {frame_path}")
                if index in (0, 48, 96):
                    cv2.imwrite(str(frames_output / f"frame_{index:03d}.png"), grid)
                    sampled_stdev["base"][str(index)] = float(decoded["base"].std())
                    sampled_stdev["dpo"][str(index)] = float(decoded["dpo"].std())
            _encode_grid_frames(frame_dir, sample.fps, comparison)
    finally:
        for capture in captures.values():
            capture.release()

    generated_nonblank = {
        name: all(value > 2.0 for value in sampled_stdev[name].values())
        for name in ("base", "dpo")
    }
    comparison_probe = _ffprobe(comparison)
    if _frame_count(comparison_probe) != sample.num_frames:
        raise ValueError("comparison grid frame count mismatch")
    conditioning_chain_valid = bool(all(generated_nonblank.values()))
    report = {
        "sample_id": sample_id,
        "status": "awaiting_human_review" if conditioning_chain_valid else "failed",
        "conditioning_chain_valid": conditioning_chain_valid,
        "mask_preprocessor": MASK_LABEL,
        "target_usage": "evaluation-only; excluded from generation conditioning",
        "fairness_check": "passed",
        "condition_hashes": condition_hashes["base"],
        "files": {
            name: {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in {**paths, "reference": sample.reference_image, "comparison": comparison}.items()
        },
        "ffprobe": {**probes, "comparison": comparison_probe},
        "sampled_frame_stdev": sampled_stdev,
        "generated_nonblank": generated_nonblank,
        "review_frames": [0, 48, 96],
        "review": {
            "identity": "pending",
            "face": "pending",
            "clothing_hair": "pending",
            "motion": "pending",
            "flicker": "pending",
            "body_edges": "pending",
            "background_artifacts": "pending",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.root.resolve(), args.sample_id, args.output.resolve())
    print(json.dumps({"sample_id": args.sample_id, "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
