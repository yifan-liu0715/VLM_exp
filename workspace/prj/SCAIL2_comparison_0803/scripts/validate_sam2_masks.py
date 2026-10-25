#!/usr/bin/env python3
"""Encode, inspect, and fail-closed validate one SAM2 fallback mask bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam2_fallback_0808.inputs import assert_conditioning_paths, load_sample
from sam2_fallback_0808.palette import encode_reference, encode_video_frame
from sam2_fallback_0808.quality import (
    clean_reference_components,
    compute_mask_stats,
    validate_human_review,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _encode_png_sequence(frame_dir: Path, fps: float, output: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv444p",
        str(output),
    ]
    subprocess.run(command, check=True)


def _encode_video_prefix(source: Path, frame_count: int, fps: float, output: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-an",
        "-r",
        f"{fps:g}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)


def _read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def _frame_count(probe: dict) -> int:
    stream = probe["streams"][0]
    value = stream.get("nb_read_frames") or stream.get("nb_frames")
    return int(value)


def _write_overlays(source: Path, masks: list[np.ndarray], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open source for overlays: {source}")
    try:
        for index in (0, 48, 96):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"cannot extract source frame {index}")
            if frame.shape[:2] != masks[index].shape:
                raise ValueError(f"overlay shape mismatch at frame {index}")
            overlay = frame.copy()
            overlay[masks[index]] = (255, 0, 0)
            blended = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)
            cv2.imwrite(str(output / f"frame_{index:03d}.png"), blended)
    finally:
        capture.release()


def _self_test_codec() -> None:
    masks = [np.zeros((32, 32), bool) for _ in range(2)]
    for mask in masks:
        mask[8:24, 8:24] = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = root / "frames"
        frames.mkdir()
        for index, mask in enumerate(masks):
            Image.fromarray(encode_video_frame(mask)).save(
                frames / f"frame_{index:05d}.png"
            )
        video = root / "roundtrip.mp4"
        _encode_png_sequence(frames, 24.0, video)
        capture = cv2.VideoCapture(str(video))
        ok, bgr = capture.read()
        capture.release()
        if not ok or bgr is None:
            raise RuntimeError("codec self-test could not decode frame")
        foreground = bgr[16, 16]
        background = bgr[0, 0]
        if not (foreground[0] > 200 and foreground[1] < 64 and foreground[2] < 64):
            raise RuntimeError(f"blue foreground failed codec round-trip: {foreground.tolist()}")
        if not np.all(background < 32):
            raise RuntimeError(f"black background failed codec round-trip: {background.tolist()}")


def validate_bundle(root: Path, sample_id: str, require_human_review: bool) -> dict:
    sample = load_sample(root, sample_id)
    assert_conditioning_paths(sample, [sample.source_video, sample.reference_image])
    output = root / "preprocessed_sam2" / sample.sample_id
    raw = output / "raw_masks"
    reference_raw = raw / "reference.png"
    video_dir = raw / "video"
    if not reference_raw.is_file():
        raise FileNotFoundError(reference_raw)
    frame_paths = [video_dir / f"frame_{index:05d}.png" for index in range(sample.num_frames)]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing raw mask frames: {missing[:4]}")

    reference_mask = clean_reference_components(_read_binary_mask(reference_raw))
    masks = [_read_binary_mask(path) for path in frame_paths]
    stats = compute_mask_stats(masks)
    if [int(v) for v in stats["shape"]] != [sample.height, sample.width]:
        raise ValueError(f"raw mask dimensions {stats['shape']} do not match source")
    if len(masks) != sample.num_frames:
        raise ValueError(f"expected {sample.num_frames} raw masks")

    review_path = output / "qc_review.json"
    human_review_passed = False
    if review_path.is_file():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        validate_human_review(review)
        human_review_passed = True
    elif require_human_review:
        raise ValueError(f"human review required: {review_path}")
    else:
        template = {
            "status": "pending",
            "frames": [0, 48, 96],
            "id_switch": None,
            "large_background_absorption": None,
        }
        (output / "qc_review_template.json").write_text(
            json.dumps(template, indent=2) + "\n", encoding="utf-8"
        )

    reference_encoded = output / "ref_mask.jpg"
    Image.fromarray(encode_reference(reference_mask)).save(
        reference_encoded, quality=100, subsampling=0
    )
    mask_video = output / "rendered_mask_v2.mp4"
    with tempfile.TemporaryDirectory(dir=output, prefix="encode_frames_") as tmp:
        encoded_frames = Path(tmp)
        for index, mask in enumerate(masks):
            Image.fromarray(encode_video_frame(mask)).save(
                encoded_frames / f"frame_{index:05d}.png"
            )
        _encode_png_sequence(encoded_frames, sample.fps, mask_video)
    driving_copy = output / "rendered_v2.mp4"
    _encode_video_prefix(sample.source_video, sample.num_frames, sample.fps, driving_copy)
    _write_overlays(sample.source_video, masks, output / "qc_overlays")

    driving_probe = _ffprobe(driving_copy)
    mask_probe = _ffprobe(mask_video)
    for name, probe in (("rendered_v2", driving_probe), ("rendered_mask_v2", mask_probe)):
        stream = probe["streams"][0]
        if (int(stream["width"]), int(stream["height"])) != (sample.width, sample.height):
            raise ValueError(f"{name} dimensions do not match sample")
        if _frame_count(probe) != sample.num_frames:
            raise ValueError(f"{name} does not have {sample.num_frames} frames")

    automatic_pass = bool(stats["automatic_pass"])
    status = "passed" if automatic_pass and human_review_passed else (
        "awaiting_human_review" if automatic_pass else "failed"
    )
    manifest = {
        "sample_id": sample.sample_id,
        "status": status,
        "mask_preprocessor": "official SAM2 fallback; not official SAM3",
        "target_usage": "evaluation-only; excluded from mask preprocessing",
        "conditioning_paths": [str(sample.source_video), str(sample.reference_image)],
        "width": sample.width,
        "height": sample.height,
        "num_frames": sample.num_frames,
        "fps": sample.fps,
        "mask_stats": stats,
        "human_review_passed": human_review_passed,
        "files": {
            path.name: {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (reference_encoded, driving_copy, mask_video)
        },
        "ffprobe": {"rendered_v2": driving_probe, "rendered_mask_v2": mask_probe},
        "review_frames": [0, 48, 96],
    }
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if str(sample.target_video) in text:
        raise ValueError("target leakage in mask manifest")
    (output / "mask_manifest.json").write_text(text + "\n", encoding="utf-8")
    if require_human_review and status != "passed":
        raise ValueError(f"mask bundle did not pass: {status}")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--require-human-review", action="store_true")
    parser.add_argument("--self-test-codec", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test_codec:
        _self_test_codec()
        print("codec self-test passed")
        return 0
    if args.root is None or not args.sample_id:
        parser.error("--root and --sample-id are required unless --self-test-codec is used")
    manifest = validate_bundle(args.root.resolve(), args.sample_id, args.require_human_review)
    print(json.dumps({"sample_id": args.sample_id, "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
