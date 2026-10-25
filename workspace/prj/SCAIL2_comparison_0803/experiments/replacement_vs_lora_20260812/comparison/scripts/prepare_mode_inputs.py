#!/usr/bin/env python3
"""Prepare one immutable 65-frame input bundle for a mode comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from mode_comparison.contracts import (
    MASK_PREPROCESSOR,
    REFERENCE_IMAGE,
    SOURCE_VIDEO,
    TARGET_VIDEO,
    ExperimentContract,
    comparison_contract,
    temporal_indices,
)
from mode_comparison.media import (
    ACTOR_RGB,
    colorize_reference_mask,
    colorize_replacement_mask,
    fit_pad_mask,
    fit_pad_rgb,
    mask_statistics,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_frame_index_record(contract: ExperimentContract) -> dict:
    indices = temporal_indices(contract.input_frames, contract.output_frames)
    samples = []
    for output_index, source_index in enumerate(indices):
        samples.append(
            {
                "output_index": output_index,
                "source_index": source_index,
                "output_time_seconds": output_index / contract.output_fps,
                "source_time_seconds": source_index / contract.input_fps,
            }
        )
    return {
        "policy": "nearest_source_frame_across_both_endpoints",
        "source_frames": contract.input_frames,
        "source_fps": contract.input_fps,
        "output_frames": contract.output_frames,
        "output_fps": contract.output_fps,
        "indices": list(indices),
        "samples": samples,
    }


def validate_raw_mask_manifest(payload: dict, contract: ExperimentContract) -> dict:
    expected = {
        "sample_id": contract.sample_id,
        "frame_count": contract.input_frames,
        "fps": float(contract.input_fps),
        "width": 704,
        "height": 1280,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"raw mask manifest {key}={payload.get(key)!r}, expected {value!r}")
    paths = list(payload.get("conditioning_paths", []))
    if paths != [SOURCE_VIDEO, REFERENCE_IMAGE]:
        raise ValueError("raw masks must be conditioned only on frozen source and reference")
    serialized = json.dumps(payload, ensure_ascii=False)
    if TARGET_VIDEO in serialized:
        raise ValueError("target leakage found in raw mask manifest")
    return {
        **payload,
        "mask_preprocessor": MASK_PREPROCESSOR,
        "source_manifest_label": payload.get("mask_preprocessor"),
    }


def prepared_paths(experiment_root: Path) -> dict[str, Path]:
    root = Path(experiment_root)
    return {
        "comparison_source": root / "comparison/inputs/source.mp4",
        "comparison_target": root / "comparison/inputs/target.mp4",
        "comparison_reference": root / "comparison/inputs/reference.png",
        "frame_indices": root / "comparison/manifests/frame_indices.json",
        "preparation_manifest": root / "comparison/manifests/input_preparation.json",
        "scail_driving": root / "scail2/inputs/driving.mp4",
        "scail_reference": root / "scail2/inputs/reference.png",
        "scail_ref_mask": root / "scail2/inputs/ref_mask.png",
        "scail_replace_mask": root / "scail2/inputs/replace_mask.mp4",
        "scail_rendered": root / "scail2/inputs/rendered_v2.mp4",
        "scail_input_manifest": root / "scail2/manifests/input_manifest.json",
        "wan_source": root / "wan_lora/inputs/source.mp4",
        "wan_target": root / "wan_lora/inputs/target.mp4",
        "wan_reference": root / "wan_lora/inputs/reference.png",
        "wan_input_manifest": root / "wan_lora/manifests/input_manifest.json",
    }


def validate_video_metadata(
    *,
    observed_frames: int,
    observed_fps: float,
    observed_width: int,
    observed_height: int,
    contract: ExperimentContract,
) -> dict[str, object]:
    if observed_frames < contract.input_frames:
        raise ValueError(
            f"video has {observed_frames} frames, need at least {contract.input_frames}"
        )
    if abs(observed_fps - contract.input_fps) > 0.01:
        raise ValueError(f"video has fps {observed_fps}, expected {contract.input_fps}")
    if (observed_width, observed_height) != (704, 1280):
        raise ValueError(
            f"video has canvas {observed_width}x{observed_height}, expected 704x1280"
        )
    return {
        "file_frames": int(observed_frames),
        "clip_frames": contract.input_frames,
        "clip_start": 0,
        "clip_end_inclusive": contract.input_frames - 1,
        "fps": float(observed_fps),
        "width": int(observed_width),
        "height": int(observed_height),
    }


def _read_video(
    path: Path,
    *,
    indices: tuple[int, ...],
    contract: ExperimentContract,
) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    observed_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    observed_fps = float(capture.get(cv2.CAP_PROP_FPS))
    observed_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    observed_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        validate_video_metadata(
            observed_frames=observed_frames,
            observed_fps=observed_fps,
            observed_width=observed_width,
            observed_height=observed_height,
            contract=contract,
        )
    except ValueError:
        capture.release()
        raise
    wanted = set(indices)
    frames: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames[frame_index] = fit_pad_rgb(
                    rgb,
                    width=contract.width,
                    height=contract.height,
                )
            frame_index += 1
    finally:
        capture.release()
    missing = [index for index in indices if index not in frames]
    if missing:
        raise ValueError(f"video is missing sampled frames: {missing[:8]}")
    return [frames[index] for index in indices]


def _read_sampled_masks(
    mask_dir: Path,
    *,
    indices: tuple[int, ...],
    contract: ExperimentContract,
) -> list[np.ndarray]:
    output = []
    for index in indices:
        path = mask_dir / f"frame_{index:05d}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as handle:
            raw = np.asarray(handle.convert("L")) > 127
        output.append(fit_pad_mask(raw, width=contract.width, height=contract.height))
    return output


def _write_video(
    frames: list[np.ndarray],
    path: Path,
    *,
    fps: int,
    ffmpeg: str,
    lossless_rgb: bool = False,
) -> None:
    if not frames:
        raise ValueError("cannot encode an empty video")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frames_", dir=path.parent) as directory:
        frame_root = Path(directory)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_root / f"frame_{index:05d}.png")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            str(frame_root / "frame_%05d.png"),
            "-frames:v",
            str(len(frames)),
        ]
        if lossless_rgb:
            command += ["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"]
        else:
            command += [
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "12",
                "-pix_fmt",
                "yuv420p",
            ]
        command += ["-r", str(fps), str(path)]
        subprocess.run(command, check=True)


def _probe_video(path: Path, *, ffprobe: str) -> dict[str, object]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream["avg_frame_rate"])),
        "frames": int(stream["nb_read_frames"]),
    }


def _assert_expected_probe(probe: dict[str, object], contract: ExperimentContract) -> None:
    expected = {
        "width": contract.width,
        "height": contract.height,
        "fps": float(contract.output_fps),
        "frames": contract.output_frames,
    }
    if probe != expected:
        raise ValueError(f"encoded media probe {probe}, expected {expected}")


def _ensure_outputs_absent(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite prepared outputs: {existing[:5]}")


def prepare(args: argparse.Namespace) -> dict:
    contract = comparison_contract(args.mode)
    if contract.mode != "replacement":
        raise ValueError("this preprocessor currently prepares replacement-mode inputs only")
    paths = prepared_paths(args.experiment_root.resolve())
    _ensure_outputs_absent(paths)

    raw_payload = json.loads(args.raw_mask_manifest.read_text(encoding="utf-8"))
    normalized_lineage = validate_raw_mask_manifest(raw_payload, contract)
    indices = temporal_indices(contract.input_frames, contract.output_frames)
    source_frames = _read_video(args.source_video, indices=indices, contract=contract)
    target_frames = _read_video(args.target_video, indices=indices, contract=contract)
    sampled_masks = _read_sampled_masks(
        args.raw_video_mask_dir,
        indices=indices,
        contract=contract,
    )
    with Image.open(args.reference_image) as handle:
        reference = fit_pad_rgb(
            np.asarray(handle.convert("RGB")),
            width=contract.width,
            height=contract.height,
        )
    with Image.open(args.raw_reference_mask) as handle:
        reference_mask = fit_pad_mask(
            np.asarray(handle.convert("L")) > 127,
            width=contract.width,
            height=contract.height,
        )

    _write_video(
        source_frames,
        paths["comparison_source"],
        fps=contract.output_fps,
        ffmpeg=args.ffmpeg,
    )
    _write_video(
        target_frames,
        paths["comparison_target"],
        fps=contract.output_fps,
        ffmpeg=args.ffmpeg,
    )
    paths["comparison_reference"].parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(reference).save(paths["comparison_reference"])
    Image.fromarray(colorize_reference_mask(reference_mask)).save(
        paths["scail_ref_mask"]
    )
    replacement_frames = [colorize_replacement_mask(mask) for mask in sampled_masks]
    _write_video(
        replacement_frames,
        paths["scail_replace_mask"],
        fps=contract.output_fps,
        ffmpeg=args.ffmpeg,
        lossless_rgb=True,
    )

    for source_key, destination_key in (
        ("comparison_source", "scail_driving"),
        ("comparison_source", "scail_rendered"),
        ("comparison_source", "wan_source"),
        ("comparison_target", "wan_target"),
        ("comparison_reference", "scail_reference"),
        ("comparison_reference", "wan_reference"),
    ):
        paths[destination_key].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[source_key], paths[destination_key])

    frame_record = build_frame_index_record(contract)
    atomic_json(paths["frame_indices"], frame_record)
    common = {
        "sample_id": contract.sample_id,
        "prompt": args.prompt,
        "width": contract.width,
        "height": contract.height,
        "num_frames": contract.output_frames,
        "fps": contract.output_fps,
        "frame_indices_manifest": str(paths["frame_indices"]),
        "target_usage": "evaluation-only; excluded from generation conditioning",
    }
    scail_manifest = {
        **common,
        "mode": "replacement",
        "replace_flag": True,
        "image": str(paths["scail_reference"]),
        "mask_image": str(paths["scail_ref_mask"]),
        "pose": str(paths["scail_rendered"]),
        "mask_video": str(paths["scail_replace_mask"]),
        "mask_preprocessor": MASK_PREPROCESSOR,
        "raw_mask_lineage": normalized_lineage,
        "actor_rgb": list(ACTOR_RGB),
        "mask_qc": {
            "reference": mask_statistics(reference_mask),
            "frames": {
                str(index): mask_statistics(sampled_masks[index])
                for index in (0, 32, 64)
            },
        },
    }
    wan_manifest = {
        **common,
        "source_video": str(paths["wan_source"]),
        "target_video": str(paths["wan_target"]),
        "reference_image": str(paths["wan_reference"]),
        "generation_conditioning": [
            str(paths["wan_source"]),
            str(paths["wan_reference"]),
            args.prompt,
        ],
    }
    atomic_json(paths["scail_input_manifest"], scail_manifest)
    atomic_json(paths["wan_input_manifest"], wan_manifest)

    probes = {}
    for key in (
        "comparison_source",
        "comparison_target",
        "scail_driving",
        "scail_rendered",
        "scail_replace_mask",
        "wan_source",
        "wan_target",
    ):
        probe = _probe_video(paths[key], ffprobe=args.ffprobe)
        _assert_expected_probe(probe, contract)
        probes[key] = probe
    file_hashes = {
        key: sha256_file(path)
        for key, path in paths.items()
        if path.is_file() and key != "preparation_manifest"
    }
    preparation_manifest = {
        "status": "passed",
        "mode": contract.mode,
        "contract": common,
        "source_files": {
            "source_video": str(args.source_video),
            "target_video": str(args.target_video),
            "reference_image": str(args.reference_image),
            "raw_reference_mask": str(args.raw_reference_mask),
            "raw_video_mask_dir": str(args.raw_video_mask_dir),
            "raw_mask_manifest": str(args.raw_mask_manifest),
        },
        "probes": probes,
        "sha256": file_hashes,
    }
    atomic_json(paths["preparation_manifest"], preparation_manifest)
    return preparation_manifest


def parse_args() -> argparse.Namespace:
    contract = comparison_contract("replacement")
    root = Path(contract.experiment_root)
    old_masks = (
        Path("/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/preprocessed_sam2")
        / contract.sample_id
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="replacement", choices=["replacement"])
    parser.add_argument("--experiment-root", type=Path, default=root)
    parser.add_argument("--source-video", type=Path, default=Path(SOURCE_VIDEO))
    parser.add_argument("--target-video", type=Path, default=Path(TARGET_VIDEO))
    parser.add_argument("--reference-image", type=Path, default=Path(REFERENCE_IMAGE))
    parser.add_argument(
        "--raw-reference-mask",
        type=Path,
        default=old_masks / "raw_masks/reference.png",
    )
    parser.add_argument(
        "--raw-video-mask-dir",
        type=Path,
        default=old_masks / "raw_masks/video",
    )
    parser.add_argument(
        "--raw-mask-manifest",
        type=Path,
        default=old_masks / "raw_mask_manifest.json",
    )
    parser.add_argument("--prompt", default=contract_prompt())
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def contract_prompt() -> str:
    from mode_comparison.contracts import PROMPT

    return PROMPT


def main() -> None:
    result = prepare(parse_args())
    print(json.dumps({"status": result["status"], "mode": result["mode"]}))


if __name__ == "__main__":
    main()
