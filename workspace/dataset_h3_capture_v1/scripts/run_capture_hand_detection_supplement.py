#!/usr/bin/env python3
"""Extract MediaPipe 21-point hand landmarks from exported capture clips."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def load_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[max(0, math.ceil(0.95 * len(values)) - 1)]


def detect_clip(
    landmarker: vision.HandLandmarker,
    clip_path: Path,
    row: dict[str, Any],
    timestamp_base_ms: int,
    sample_fps: float,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {clip_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = float(row.get("output", {}).get("duration_sec") or row.get("duration_sec") or 0.0)
    if duration <= 0 and frame_count > 0:
        duration = frame_count / source_fps

    frames: list[dict[str, Any]] = []
    next_sample = 0.0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            time_sec = frame_index / source_fps
            frame_index += 1
            if time_sec + 1e-6 < next_sample:
                continue
            if duration > 0 and time_sec >= duration + 1e-3:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = timestamp_base_ms + int(round(time_sec * 1000.0))
            result = landmarker.detect_for_video(image, timestamp_ms)
            hands: list[dict[str, Any]] = []
            for hand_index, landmarks in enumerate(result.hand_landmarks):
                handedness = None
                handedness_score = None
                if hand_index < len(result.handedness) and result.handedness[hand_index]:
                    category = result.handedness[hand_index][0]
                    handedness = category.category_name or category.display_name
                    handedness_score = category.score
                hands.append({
                    "handedness": handedness,
                    "handedness_score": handedness_score,
                    "points": [[float(point.x), float(point.y), float(point.z)] for point in landmarks],
                })
            frames.append({
                "time_sec": round(time_sec, 4),
                "hands": hands,
                "hand_count": len(hands),
            })
            next_sample += 1.0 / sample_fps
    finally:
        cap.release()

    counts = [frame["hand_count"] for frame in frames]
    visible = sum(count > 0 for count in counts)
    two_hands = sum(count >= 2 for count in counts)
    centers: list[tuple[float, float]] = []
    for frame in frames:
        if frame["hands"]:
            points = frame["hands"][0]["points"]
            ids = (0, 5, 9, 13, 17)
            centers.append((
                sum(points[index][0] for index in ids) / len(ids),
                sum(points[index][1] for index in ids) / len(ids),
            ))
    center_steps = [
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(centers, centers[1:])
    ]
    summary = dict(row)
    summary.update({
        "clip_path": str(clip_path),
        "keypoints_extracted": True,
        "frames_sampled": len(frames),
        "sample_fps": sample_fps,
        "hand_visible_rate": visible / len(frames) if frames else 0.0,
        "two_hand_rate": two_hands / len(frames) if frames else 0.0,
        "mean_center_step": sum(center_steps) / len(center_steps) if center_steps else 0.0,
        "p95_center_step": _p95(center_steps),
        "landmarker": "MediaPipe Hand Landmarker 0.10.21",
    })
    return {"summary": summary, "frames": frames}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-fps", type=float, default=24.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0,
                        help="Exclusive row index; 0 means the end of the manifest.")
    parser.add_argument("--opencv-threads", type=int, default=1)
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")

    cv2.setNumThreads(max(1, args.opencv_threads))
    rows = load_manifest(Path(args.manifest))
    start = max(0, args.start)
    end = len(rows) if args.end <= 0 else min(len(rows), args.end)
    rows = rows[start:end]
    clip_dir = Path(args.clip_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Path(args.model)
    if not model.is_file():
        raise FileNotFoundError(model)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for local_index, row in enumerate(rows):
            clip_id = str(row["clip_id"])
            output_path = out_dir / f"{clip_id}_hands.json"
            if output_path.is_file() and output_path.stat().st_size > 100:
                try:
                    summaries.append(json.loads(output_path.read_text(encoding="utf-8"))["summary"])
                    print(f"SKIP {clip_id}", flush=True)
                    continue
                except (OSError, UnicodeError, json.JSONDecodeError, KeyError):
                    pass
            clip_name = Path(str(row.get("output_video", f"{clip_id}.mp4"))).name
            clip_path = clip_dir / clip_name
            if not clip_path.is_file():
                error = {"clip_id": clip_id, "error": f"missing clip: {clip_path}"}
                errors.append(error)
                print(f"ERROR {clip_id} {error['error']}", file=sys.stderr, flush=True)
                continue
            try:
                payload = detect_clip(
                    landmarker,
                    clip_path,
                    row,
                    timestamp_base_ms=(start + local_index + 1) * 10_000_000,
                    sample_fps=args.sample_fps,
                )
                output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                summaries.append(payload["summary"])
                print(
                    f"[{local_index + 1}/{len(rows)}] {clip_id} "
                    f"frames={payload['summary']['frames_sampled']} "
                    f"visible={payload['summary']['hand_visible_rate']:.3f}",
                    flush=True,
                )
            except Exception as exc:  # keep the batch resumable
                error = {"clip_id": clip_id, "error": repr(exc)}
                errors.append(error)
                print(f"ERROR {clip_id} {error['error']}", file=sys.stderr, flush=True)

    summaries.sort(key=lambda item: item["clip_id"])
    report = {
        "manifest_count": len(rows),
        "completed_count": len(summaries),
        "error_count": len(errors),
        "errors": errors,
        "samples": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("manifest_count", "completed_count", "error_count")}), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
