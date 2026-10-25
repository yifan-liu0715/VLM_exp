#!/usr/bin/env python3
"""Audit completed Scheme 4 comparison videos and manifests."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


REQUIRED_VIDEO_NAMES = (
    "Source.mp4",
    "Target.mp4",
    "Base.mp4",
    "Scheme3.mp4",
    "Scheme4.mp4",
    "comparison_base_scheme4_4grid.mp4",
    "comparison_scheme3_scheme4_4grid.mp4",
)
SAMPLE_IDS = (
    "mymotion__051",
    "mymotion__single_talking_face__datatrain__002",
    "mymotion__humandance__batch1_mingxing_cropped__002",
)


def expected_sample_outputs(root: Path, sample_id: str) -> dict:
    sample_root = Path(root) / sample_id
    return {
        "videos": {
            name: sample_root / "videos" / name
            for name in REQUIRED_VIDEO_NAMES
        },
        "manifest": sample_root / "run_manifest.json",
    }


def validate_video_probe(payload: dict) -> dict:
    video_streams = [
        stream
        for stream in payload.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise RuntimeError(
            f"expected one video stream, found {len(video_streams)}"
        )
    stream = video_streams[0]
    frame_count = int(stream.get("nb_read_frames", -1))
    if frame_count != 97:
        raise RuntimeError(f"expected 97 frames, found {frame_count}")
    frame_rate = Fraction(str(stream.get("avg_frame_rate", "0/1")))
    if frame_rate != Fraction(24, 1):
        raise RuntimeError(f"expected 24 fps, found {frame_rate}")
    return {
        "frames": frame_count,
        "fps": float(frame_rate),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream.get("codec_name"),
    }


def probe_video(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return validate_video_probe(json.loads(completed.stdout))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {"status": "passed", "root": str(args.root), "samples": []}
    for sample_id in SAMPLE_IDS:
        expected = expected_sample_outputs(args.root, sample_id)
        manifest_path = expected["manifest"]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_outputs = {
            Path(path).resolve()
            for path in manifest.get("outputs", {}).values()
        }
        videos = {}
        for name, path in expected["videos"].items():
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"missing or empty video: {path}")
            if path.resolve() not in manifest_outputs:
                raise RuntimeError(f"{name} is absent from {manifest_path}")
            videos[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "probe": probe_video(path),
            }
        report["samples"].append(
            {
                "sample_id": sample_id,
                "manifest": str(manifest_path),
                "videos": videos,
            }
        )
    output = args.output or (args.root / "audit_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
