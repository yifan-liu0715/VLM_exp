#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT = Path("/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio")
DATASET = PROJECT / "data/dataset_video_train"
METADATA = DATASET / "metadata_train.csv"
OUT = PROJECT / "data_0713/v2v_quick_0713"
TEST_IDS = ["mymotion__002", "mymotion__012", "mymotion__015"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare isolated target-domain training data and deterministic V2V evaluation pairs.")
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--overwrite-sources", action="store_true")
    return parser.parse_args()


def absolute(entry: str) -> Path:
    path = Path(entry)
    return path if path.is_absolute() else DATASET / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ffmpeg(source: Path, destination: Path, vf: str, crf: int):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", vf,
        "-an", "-frames:v", "33", "-r", "15",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with METADATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_id = {Path(row["video"]).stem: row for row in rows}
    missing = [sample_id for sample_id in TEST_IDS if sample_id not in by_id]
    if missing:
        raise ValueError(f"Missing test IDs: {missing}")

    candidates = []
    for row in rows:
        sample_id = Path(row["video"]).stem
        video = absolute(row["video"])
        if sample_id in TEST_IDS or not video.is_file():
            continue
        if "1280x704" not in video.parts:
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: Path(row["video"]).stem)
    train_rows = candidates[: args.train_limit]
    if len(train_rows) < args.train_limit:
        raise RuntimeError(f"Only {len(train_rows)} eligible rows found; requested {args.train_limit}")

    train_csv = OUT / "train_targets_0713.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
        writer.writeheader()
        for row in train_rows:
            writer.writerow({"video": row["video"], "prompt": row["prompt"]})

    degradation = (
        "fps=15,trim=end_frame=33,setpts=PTS-STARTPTS,"
        "scale=288:160:flags=area,gblur=sigma=1.2,"
        "scale=576:320:flags=bicubic,format=yuv420p"
    )
    target_filter = "fps=15,trim=end_frame=33,setpts=PTS-STARTPTS,scale=576:320:flags=lanczos,format=yuv420p"
    pair_rows = []
    for sample_id in TEST_IDS:
        row = by_id[sample_id]
        original = absolute(row["video"])
        if not original.is_file():
            raise FileNotFoundError(original)
        source = OUT / "source_videos" / f"{sample_id}_source_degraded.mp4"
        target = OUT / "target_videos" / f"{sample_id}_target.mp4"
        if args.overwrite_sources or not source.is_file():
            run_ffmpeg(original, source, degradation, crf=32)
        if args.overwrite_sources or not target.is_file():
            run_ffmpeg(original, target, target_filter, crf=17)
        pair_rows.append({
            "sample_id": sample_id,
            "split": "heldout_v2v_test",
            "source_video": str(source),
            "target_video": str(target),
            "original_video": str(original),
            "reference_image": str(absolute(row["reference_image"])),
            "prompt": row["prompt"],
            "source_recipe": degradation + "; libx264 crf=32",
            "source_sha256": sha256(source),
            "target_sha256": sha256(target),
        })

    pair_csv = OUT / "v2v_pairs_0713.csv"
    with pair_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    manifest = {
        "experiment": "wan2.2_ti2v5b_native_input_video_v2v_lora_0713",
        "method": "Target-domain LoRA training; native input_video V2V inference with deterministic degraded sources.",
        "important_limitation": "The plain TI2V-5B architecture has no separate trainable source-video conditioning branch. Source videos are used by the native V2V latent initialization at inference, not as an extra paired condition during LoRA training.",
        "source_metadata": str(METADATA),
        "train_csv": str(train_csv),
        "train_rows": len(train_rows),
        "train_sample_ids": [Path(row["video"]).stem for row in train_rows],
        "heldout_sample_ids": TEST_IDS,
        "pair_csv": str(pair_csv),
        "resolution": {"height": 320, "width": 576, "num_frames": 33, "fps": 15},
        "source_recipe": degradation + "; libx264 crf=32",
    }
    (OUT / "data_manifest_0713.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train_rows": len(train_rows), "train_csv": str(train_csv), "pair_csv": str(pair_csv)}, indent=2))


if __name__ == "__main__":
    main()
