#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/dataset_h3_capture_manual_reviewed_1061_keypoints_20260928
BASE=/nvmedata/workspace2/users/nina/dataset_h3_capture_semantic_rescreen_v3_2016_coverage357_20260923
STAGE=/nvmedata/workspace2/users/nina/.staging_h3_manual_reviewed_1061_20260928
PYTHON=/nvmedata/workspace2/conda_data/envs/v2v_train/bin/python
SOURCE_MODEL=/nvmedata/workspace2/users/nina/dataset_h3_public_v1/models/hand_landmarker.task
MANIFEST_SHA=556c4395c79046d17bfc70edd5fa250a05d2bb3455ff88cf8afd43df19e9b601
EXPECTED=1061
WORKERS=12

mkdir -p "$ROOT"/{clips,hand_keypoints,hand_keypoints_shards,logs,manifests,models,reports,scripts}
if [[ -e "$ROOT/.manual_reviewed_1061_marker" ]]; then
  grep -qx 'h3-manual-reviewed-1061-20260928' "$ROOT/.manual_reviewed_1061_marker"
else
  printf '%s\n' 'h3-manual-reviewed-1061-20260928' >"$ROOT/.manual_reviewed_1061_marker"
fi

cp -p "$STAGE/manual_reviewed_1061_hand_manifest.jsonl" "$ROOT/manifests/manual_reviewed_1061.jsonl"
actual_manifest_sha=$(sha256sum "$ROOT/manifests/manual_reviewed_1061.jsonl" | cut -d' ' -f1)
[[ "$actual_manifest_sha" == "$MANIFEST_SHA" ]]

for name in run_capture_hand_detection.py run_sharded_hand_detection.py merge_hand_keypoint_shards.py stamp_hand_keypoint_metadata.py causal_clip_pipeline.py validate_remote_causal_derivatives.py; do
  cp -p "$STAGE/$name" "$ROOT/scripts/$name"
done
cp -p "$SOURCE_MODEL" "$ROOT/models/hand_landmarker.task"

echo "[$(date -Is)] phase=materialize start"
"$PYTHON" - "$BASE" "$ROOT" "$EXPECTED" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

base, root = map(Path, sys.argv[1:3])
expected = int(sys.argv[3])
manifest = root / "manifests/manual_reviewed_1061.jsonl"
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
if len(rows) != expected:
    raise SystemExit(f"manifest_count_expected_{expected}_actual_{len(rows)}")
ids = [str(row.get("clip_id") or "") for row in rows]
names = [str(row.get("clip_file") or "") for row in rows]
if len(set(ids)) != expected or len(set(names)) != expected or any(not value for value in ids):
    raise SystemExit("manifest_has_empty_or_duplicate_ids")

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

linked = 0
total_size = 0
for row in rows:
    clip_id = str(row["clip_id"])
    name = str(row["clip_file"])
    if Path(name).name != name or Path(name).stem != clip_id:
        raise SystemExit(f"unsafe_or_mismatched_filename:{name}")
    source = base / "clips" / name
    target = root / "clips" / name
    if not source.is_file():
        raise SystemExit(f"missing_base_clip:{source}")
    actual = digest(source)
    if actual != row.get("output_sha256"):
        raise SystemExit(f"base_sha_mismatch:{clip_id}")
    if source.stat().st_size != int(row.get("output_size_bytes") or -1):
        raise SystemExit(f"base_size_mismatch:{clip_id}")
    if target.exists():
        if not target.is_file() or not os.path.samefile(source, target):
            raise SystemExit(f"existing_target_is_not_expected_hardlink:{target}")
    else:
        os.link(source, target)
    if not os.path.samefile(source, target):
        raise SystemExit(f"hardlink_verification_failed:{clip_id}")
    linked += 1
    total_size += target.stat().st_size

expected_names = set(names)
actual_names = {path.name for path in (root / "clips").glob("*.mp4") if path.is_file()}
if actual_names != expected_names:
    raise SystemExit("target_clip_set_mismatch")
report = {
    "manifest_count": len(rows),
    "hardlinked_count": linked,
    "source_coverage_count": len({str(row.get("source_sha256") or "") for row in rows}),
    "total_size_bytes": total_size,
    "local_sha_match_count": linked,
    "materialization_mode": "verified_hardlink_from_semantic_rescreen_v3",
}
(root / "reports/materialization.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
PY
echo "[$(date -Is)] phase=materialize complete"

echo "[$(date -Is)] phase=detect start workers=$WORKERS"
env CUDA_VISIBLE_DEVICES= OPENCV_FFMPEG_READ_ATTEMPTS=20000 OPENCV_FFMPEG_DECODE_ATTEMPTS=20000 \
  "$PYTHON" "$ROOT/scripts/run_sharded_hand_detection.py" \
  --manifest "$ROOT/manifests/manual_reviewed_1061.jsonl" \
  --clip-dir "$ROOT/clips" \
  --shard-root "$ROOT/hand_keypoints_shards" \
  --model "$ROOT/models/hand_landmarker.task" \
  --detector "$ROOT/scripts/run_capture_hand_detection.py" \
  --python "$PYTHON" \
  --workers "$WORKERS" \
  --sample-fps 24 \
  >"$ROOT/logs/sharded_detection.log" 2>&1
echo "[$(date -Is)] phase=detect complete"

echo "[$(date -Is)] phase=merge_stamp_validate start"
"$PYTHON" "$ROOT/scripts/merge_hand_keypoint_shards.py" \
  --manifest "$ROOT/manifests/manual_reviewed_1061.jsonl" \
  --shard-root "$ROOT/hand_keypoints_shards" \
  --output "$ROOT/hand_keypoints" \
  --report "$ROOT/reports/keypoint_merge.json" \
  >"$ROOT/logs/merge_keypoints.log" 2>&1

"$PYTHON" "$ROOT/scripts/stamp_hand_keypoint_metadata.py" \
  --manifest "$ROOT/manifests/manual_reviewed_1061.jsonl" \
  --clip-dir "$ROOT/clips" \
  --json-dir "$ROOT/hand_keypoints" \
  --model "$ROOT/models/hand_landmarker.task" \
  --detector "$ROOT/scripts/run_capture_hand_detection.py" \
  --mediapipe-version 0.10.21 \
  --output-manifest "$ROOT/manifests/hand_detection_manifest_1061.jsonl" \
  >"$ROOT/logs/stamp_keypoints.log" 2>&1

"$PYTHON" "$ROOT/scripts/validate_remote_causal_derivatives.py" \
  --manifest "$ROOT/manifests/hand_detection_manifest_1061.jsonl" \
  --clip-dir "$ROOT/clips" \
  --keypoint-dir "$ROOT/hand_keypoints" \
  --report "$ROOT/reports/keypoint_validation.json" \
  >"$ROOT/logs/validate_keypoints.log" 2>&1

"$PYTHON" - "$ROOT" "$EXPECTED" <<'PY'
import datetime
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
manifest_path = root / "manifests/hand_detection_manifest_1061.jsonl"
rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
clips = sorted((root / "clips").glob("*.mp4"))
keypoints = sorted((root / "hand_keypoints").glob("*_hands.json"))
validation = json.loads((root / "reports/keypoint_validation.json").read_text(encoding="utf-8"))
merge = json.loads((root / "reports/keypoint_merge.json").read_text(encoding="utf-8"))
sharded = json.loads((root / "hand_keypoints_shards/sharded_detection_report.json").read_text(encoding="utf-8"))

def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

errors = []
ids = {str(row["clip_id"]) for row in rows}
if len(rows) != expected or len(ids) != expected:
    errors.append("manifest_count_or_id_mismatch")
if {path.stem for path in clips} != ids:
    errors.append("clip_set_mismatch")
if {path.name.removesuffix("_hands.json") for path in keypoints} != ids:
    errors.append("keypoint_set_mismatch")
if validation.get("error_count") != 0 or validation.get("valid_count") != expected:
    errors.append("keypoint_validation_failed")
if any(merge.get(key) for key in ("missing_count", "duplicate_count", "unexpected_count")):
    errors.append("keypoint_merge_failed")
if sharded.get("failure_count") != 0 or sharded.get("success_count") != sharded.get("worker_count"):
    errors.append("sharded_detection_failed")

visible_rates = []
two_hand_rates = []
frame_counts = []
zero_hand = []
for path in keypoints:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    frames = payload["frames"]
    visible = float(summary.get("hand_visible_rate") or 0.0)
    two = float(summary.get("two_hand_rate") or 0.0)
    visible_rates.append(visible)
    two_hand_rates.append(two)
    frame_counts.append(len(frames))
    if all(int(frame.get("hand_count") or 0) == 0 for frame in frames):
        zero_hand.append(str(summary["clip_id"]))

report = {
    "schema_version": 1,
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "hand_keypoints_complete" if not errors else "failed",
    "dataset_root": str(root),
    "clip_count": len(clips),
    "manifest_count": len(rows),
    "keypoint_json_count": len(keypoints),
    "source_coverage_count": len({str(row.get("source_sha256") or "") for row in rows}),
    "total_clip_size_bytes": sum(path.stat().st_size for path in clips),
    "total_keypoint_size_bytes": sum(path.stat().st_size for path in keypoints),
    "total_frames_sampled": sum(frame_counts),
    "hand_visible_ge_30_count": sum(value >= 0.30 for value in visible_rates),
    "hand_visible_ge_50_count": sum(value >= 0.50 for value in visible_rates),
    "hand_visible_ge_70_count": sum(value >= 0.70 for value in visible_rates),
    "two_hand_ge_50_count": sum(value >= 0.50 for value in two_hand_rates),
    "zero_hand_count": len(zero_hand),
    "zero_hand_clip_ids": zero_hand,
    "hand_keypoint_usable_count": expected - len(zero_hand),
    "zero_hand_training_status": "exclude_pending_manual_review" if zero_hand else "none",
    "mediapipe_version": "0.10.21",
    "sample_fps": 24.0,
    "landmarks_per_hand": 21,
    "coordinates_per_landmark": 3,
    "manifest_sha256": digest(manifest_path),
    "hand_detection_run": True,
    "training_run": False,
    "errors": errors,
}
temporary = root / "reports/.final_hand_keypoint_audit.json.tmp"
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, root / "reports/final_hand_keypoint_audit.json")
print(json.dumps(report, ensure_ascii=False))
if errors:
    raise SystemExit(2)
PY

sha256sum \
  "$ROOT/manifests/manual_reviewed_1061.jsonl" \
  "$ROOT/manifests/hand_detection_manifest_1061.jsonl" \
  "$ROOT/models/hand_landmarker.task" \
  "$ROOT/scripts/run_capture_hand_detection.py" \
  >"$ROOT/reports/provenance_sha256.txt"
printf '%s\n' 'hand_keypoints_complete_no_training' >"$ROOT/READY"
df -h /nvmedata >"$ROOT/reports/disk_after_detection.txt"
echo "[$(date -Is)] phase=merge_stamp_validate complete"
echo "[$(date -Is)] status=complete root=$ROOT"
