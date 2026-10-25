#!/usr/bin/env python3
"""Extract hand landmarks for missing EPIC pilot windows via HTTP range clips.

This is a one-shot fallback for source videos that are still downloading. It
downloads only each missing annotation window, then writes the same JSON
schema as run_epic_hand_detection.py while retaining source-time timestamps.
"""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def load_manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def detect_clip(landmarker, clip_path, row, source_index, sample_fps, time_offset):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open extracted clip {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or row.get("source_fps") or 50.0
    duration = max(0.0, float(row["clip_end_sec"]) - float(row["clip_start_sec"]))
    next_sample = 0.0
    frame_index = 0
    frames = []
    while next_sample < duration - 1e-6:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_index / fps
        frame_index += 1
        if t + 1e-6 < next_sample:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = source_index * 1_000_000_000 + int(round(t * 1000.0))
        result = landmarker.detect_for_video(image, timestamp_ms)
        hands = []
        for hi, lms in enumerate(result.hand_landmarks):
            handedness, score = None, None
            if hi < len(result.handedness) and result.handedness[hi]:
                cat = result.handedness[hi][0]
                handedness, score = cat.category_name or cat.display_name, cat.score
            hands.append({
                "handedness": handedness,
                "handedness_score": score,
                "points": [[p.x, p.y, p.z] for p in lms],
            })
        frames.append({
            "time_sec": round(time_offset + t, 4),
            "hands": hands,
            "hand_count": len(hands),
        })
        next_sample += 1.0 / sample_fps
    cap.release()
    counts = [f["hand_count"] for f in frames]
    visible = sum(c > 0 for c in counts)
    two = sum(c >= 2 for c in counts)
    centers = []
    for f in frames:
        if f["hands"]:
            pts = f["hands"][0]["points"]
            ids = [0, 5, 9, 13, 17]
            centers.append((sum(pts[i][0] for i in ids) / len(ids),
                            sum(pts[i][1] for i in ids) / len(ids)))
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(centers, centers[1:])]
    summary = dict(row)
    summary.update({
        "keypoints_extracted": True,
        "frames_sampled": len(frames),
        "sample_fps": sample_fps,
        "hand_visible_rate": visible / len(frames) if frames else 0.0,
        "two_hand_rate": two / len(frames) if frames else 0.0,
        "mean_center_step": sum(steps) / len(steps) if steps else 0.0,
        "p95_center_step": sorted(steps)[max(0, int(.95 * len(steps)) - 1)] if steps else 0.0,
        "landmarker": "MediaPipe Hand Landmarker 1.0.1",
        "source_access": "http_range_clip",
    })
    return {"summary": summary, "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-dir", default=None,
                    help="directory containing predownloaded {sample_id}.mp4 clips")
    ap.add_argument("--clip-time-base", choices=("action", "window"), default="action",
                    help="timestamp base for clips in --clip-dir")
    ap.add_argument("--no-source-fallback", action="store_true",
                    help="skip samples absent from the mirror instead of reading full source URLs")
    ap.add_argument("--sample-fps", type=float, default=24.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clip_dir = Path(args.clip_dir) if args.clip_dir else None
    if clip_dir:
        clip_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)
    pending = [r for r in rows if not (out / f"{r['sample_id']}_hands.json").exists()]
    if not pending:
        print("no missing samples", flush=True)
        return
    model = str(Path(args.manifest).parents[1] / "models" / "hand_landmarker.task")
    opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    with tempfile.TemporaryDirectory(prefix="epic_missing_clips_") as temp_dir:
        with vision.HandLandmarker.create_from_options(opts) as landmarker:
            for source_index, row in enumerate(sorted(pending, key=lambda x: x["sample_id"])):
                target = out / f"{row['sample_id']}_hands.json"
                clip = ((clip_dir / f"{row['sample_id']}.mp4") if clip_dir
                        else Path(temp_dir) / f"{row['sample_id']}.mp4")
                duration = float(row["clip_end_sec"]) - float(row["clip_start_sec"])
                if not clip.exists() or clip.stat().st_size < 1000:
                    if clip_dir:
                        mirror_url = (
                            "https://hf-mirror.com/datasets/lightly-ai/"
                            f"epic-kitchens-100-clips/resolve/main/clips/"
                            f"{row['participant_id']}/{row['sample_id']}.mp4"
                        )
                        curl_cmd = [
                            "curl", "-L", "--fail", "--retry", "8",
                            "--retry-all-errors", "--retry-delay", "3",
                            "--connect-timeout", "30", "--max-time", "900",
                            "-o", str(clip), mirror_url,
                        ]
                        print(f"downloading mirror clip {row['sample_id']} ({duration:.2f}s)", flush=True)
                        try:
                            subprocess.run(curl_cmd, check=True, timeout=1000)
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                            clip.unlink(missing_ok=True)
                    if (not clip.exists() or clip.stat().st_size < 1000) and args.no_source_fallback:
                        print(f"mirror clip unavailable; skipping {row['sample_id']}", flush=True)
                        continue
                    if not clip.exists() or clip.stat().st_size < 1000:
                        cmd = [
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-ss", str(row["clip_start_sec"]), "-i", row["download_url"],
                            "-t", str(duration), "-an", "-c:v", "copy", str(clip),
                        ]
                        print(f"downloading source clip {row['sample_id']} ({duration:.2f}s)", flush=True)
                        subprocess.run(cmd, check=True, timeout=900)
                using_action_clip = bool(clip_dir and args.clip_time_base == "action")
                time_offset = (float(row["action_start_sec"]) if using_action_clip
                               else float(row["clip_start_sec"]))
                payload = detect_clip(landmarker, clip, row, source_index, args.sample_fps,
                                      time_offset)
                payload["summary"].update({
                    "source_access": "hf_action_clip" if using_action_clip else "source_clip",
                    "available_start_sec": time_offset,
                    "available_end_sec": round(time_offset + (payload["frames"][-1]["time_sec"] - time_offset
                                                               if payload["frames"] else 0.0), 4),
                    "window_complete": not using_action_clip,
                })
                target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    summaries = []
    for p in sorted(out.glob("*_hands.json")):
        try:
            summaries.append(json.loads(p.read_text(encoding="utf-8"))["summary"])
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    (out / "summary.json").write_text(json.dumps(sorted(summaries, key=lambda x: x["sample_id"]), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"completed={len(summaries)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
