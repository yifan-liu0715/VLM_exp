#!/usr/bin/env bash
set -euo pipefail

OLD=/nvmedata/workspace2/users/nina/dataset_h3_capture_semantic_review_2050_coverage355_20260923
ROOT=/nvmedata/workspace2/users/nina/dataset_h3_capture_semantic_rescreen_v2_2045_coverage357_20260923
SOURCE_ROOT=/mnt/highspeed/users/nina/dataset_h3_capture_causal_full_20260920/sources
EXPECTED_CANDIDATE_SHA=5928ccb85cfc98ddfe13cc426d4a96690f3cbc3544b1aaf33d9a905fb9ab444b

mkdir -p "$ROOT"/{clips,logs,manifests,reports,scripts,state/exports}
printf '%s\n' 'semantic_rescreen_v2_20260923' >"$ROOT/.rescreen_v2_marker"

for name in source_inventory_portable.jsonl hand_presence_proposals.jsonl proposals_v3_all.jsonl causal_proposals_high_recall.jsonl semantic_split_decisions.jsonl; do
  cp -p "$OLD/manifests/$name" "$ROOT/manifests/$name"
done

python3 "$ROOT/scripts/build_semantic_review_candidates.py" \
  --inventory "$ROOT/manifests/source_inventory_portable.jsonl" \
  --presence "$ROOT/manifests/hand_presence_proposals.jsonl" \
  --v3 "$ROOT/manifests/proposals_v3_all.jsonl" \
  --high-recall "$ROOT/manifests/causal_proposals_high_recall.jsonl" \
  --decisions "$ROOT/manifests/semantic_split_decisions_rescreen_v2.jsonl" \
  --exclude-ids "$ROOT/manifests/semantic_rescreen_exclude_ids_v2.txt" \
  --work-root "$ROOT" \
  --target-count 2045 \
  >"$ROOT/logs/build_candidates.log" 2>&1

cp -p "$ROOT/manifests/semantic_review_candidates_2050.jsonl" "$ROOT/manifests/semantic_review_candidates_2045.jsonl"
cp -p "$ROOT/reports/semantic_review_candidates_2050.json" "$ROOT/reports/semantic_review_candidates_2045.json"

actual_candidate_sha=$(sha256sum "$ROOT/manifests/semantic_review_candidates_2045.jsonl" | cut -d' ' -f1)
if [[ "$actual_candidate_sha" != "$EXPECTED_CANDIDATE_SHA" ]]; then
  echo "candidate manifest SHA mismatch: $actual_candidate_sha" >&2
  exit 2
fi

python3 "$ROOT/scripts/audit_semantic_manifest.py" \
  --manifest "$ROOT/manifests/semantic_review_candidates_2045.jsonl" \
  --report "$ROOT/reports/semantic_candidates_independent_audit.json" \
  >"$ROOT/logs/audit_candidates.log" 2>&1

python3 "$ROOT/scripts/causal_clip_pipeline.py" prepare \
  --proposals "$ROOT/manifests/semantic_review_candidates_2045.jsonl" \
  --work-root "$ROOT" \
  --source-root "$SOURCE_ROOT" \
  --max-duration 20 \
  >"$ROOT/logs/prepare.log" 2>&1

python3 - "$OLD" "$ROOT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

old_root, root = map(Path, sys.argv[1:])

def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

old_rows = {row["clip_id"]: row for row in load(old_root / "manifests/export_manifest.jsonl")}
new_rows = load(root / "manifests/causal_segments.jsonl")
linked = 0
new_only = []
for row in new_rows:
    clip_id = row["clip_id"]
    old = old_rows.get(clip_id)
    if old is None:
        new_only.append(clip_id)
        continue
    if row["boundary_signature"] != old["boundary_signature"] or row["phase_signature"] != old["phase_signature"]:
        raise SystemExit(f"signature mismatch for reusable clip {clip_id}")
    source_video = Path(old["output_video"])
    source_sidecar = old_root / "state/exports" / f"{clip_id}.json"
    target_video = root / "clips" / f"{clip_id}.mp4"
    target_sidecar = root / "state/exports" / f"{clip_id}.json"
    if target_video.exists():
        if digest(target_video) != old["output_sha256"]:
            raise SystemExit(f"existing target video differs for {clip_id}")
    else:
        os.link(source_video, target_video)
    if target_sidecar.exists():
        if target_sidecar.read_bytes() != source_sidecar.read_bytes():
            raise SystemExit(f"existing target sidecar differs for {clip_id}")
    else:
        os.link(source_sidecar, target_sidecar)
    linked += 1

report = {
    "input_count": len(new_rows),
    "linked_existing_count": linked,
    "new_export_count": len(new_only),
    "new_clip_ids": sorted(new_only),
    "link_mode": "hardlink",
}
if linked != 2031 or len(new_only) != 14:
    raise SystemExit(f"unexpected reuse split: {report}")
(root / "reports/reuse_old_exports.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
PY

python3 "$ROOT/scripts/parallel_export_causal.py" \
  --manifest "$ROOT/manifests/causal_segments.jsonl" \
  --work-root "$ROOT" \
  --source-root "$SOURCE_ROOT" \
  --workers 4 \
  >"$ROOT/logs/export_full_2045.log" 2>&1

python3 "$ROOT/scripts/causal_clip_pipeline.py" validate \
  --manifest "$ROOT/manifests/export_manifest.jsonl" \
  --work-root "$ROOT" \
  --source-root "$SOURCE_ROOT" \
  >"$ROOT/logs/validate_full_2045.log" 2>&1

python3 "$ROOT/scripts/audit_semantic_exports.py" \
  --manifest "$ROOT/manifests/export_manifest.jsonl" \
  --work-root "$ROOT" \
  --report "$ROOT/reports/final_export_independent_audit.json" \
  >"$ROOT/logs/audit_final_exports.log" 2>&1

python3 - "$OLD" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

old_root, root = map(Path, sys.argv[1:])

def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

old_rows = {row["candidate_id"]: row for row in load(old_root / "manifests/semantic_review_candidates_2050.jsonl")}
new_rows = {row["candidate_id"]: row for row in load(root / "manifests/semantic_review_candidates_2045.jsonl")}
exports = load(root / "manifests/export_manifest.jsonl")
added_ids = sorted(set(new_rows) - set(old_rows))
removed_ids = sorted(set(old_rows) - set(new_rows))
added = [new_rows[clip_id] for clip_id in added_ids]
output_hashes = [row.get("output_sha256") for row in exports]
report = {
    "old_candidate_count": len(old_rows),
    "final_candidate_count": len(new_rows),
    "added_count": len(added_ids),
    "removed_count": len(removed_ids),
    "common_count": len(set(old_rows) & set(new_rows)),
    "source_coverage_count": len({row["source_sha256"] for row in new_rows.values()}),
    "rescreen_batch_addition_count": sum(row.get("rescreen_batch_id") == "semantic_rescreen_v2_20260923" for row in added),
    "manual_presence_override_addition_count": sum(row.get("hand_presence_override_applied") is True for row in added),
    "visual_review_artifact_addition_count": sum(bool(row.get("visual_review_artifact")) for row in added),
    "pending_user_review_count": sum(row.get("review_status") == "pending_user_review" for row in new_rows.values()),
    "semantic_feedback_verified_nonfalse_count": sum(row.get("semantic_feedback_verified") is not False for row in new_rows.values()),
    "export_count": len(exports),
    "export_ok_count": sum(row.get("export_status") == "ok" for row in exports),
    "unique_output_sha256_count": len(set(output_hashes)),
    "total_duration_sec": round(sum(float(row["duration_sec"]) for row in exports), 6),
    "total_size_bytes": sum(Path(row["output_video"]).stat().st_size for row in exports),
    "hand_keypoint_file_count": len(list((root / "hand_keypoints").glob("*_hands.json"))) if (root / "hand_keypoints").is_dir() else 0,
    "training_run": False,
    "added_ids": added_ids,
    "removed_ids": removed_ids,
}
expected = {
    "old_candidate_count": 2050,
    "final_candidate_count": 2045,
    "added_count": 14,
    "removed_count": 19,
    "common_count": 2031,
    "source_coverage_count": 357,
    "rescreen_batch_addition_count": 14,
    "manual_presence_override_addition_count": 14,
    "visual_review_artifact_addition_count": 14,
    "pending_user_review_count": 2045,
    "semantic_feedback_verified_nonfalse_count": 0,
    "export_count": 2045,
    "export_ok_count": 2045,
    "unique_output_sha256_count": 2045,
    "hand_keypoint_file_count": 0,
}
errors = {key: {"expected": value, "actual": report.get(key)} for key, value in expected.items() if report.get(key) != value}
report["errors"] = errors
(root / "reports/rescreen_v2_final_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key not in {"added_ids", "removed_ids"}}, ensure_ascii=False))
if errors:
    raise SystemExit(2)
PY

sha256sum \
  "$ROOT/manifests/semantic_review_candidates_2045.jsonl" \
  "$ROOT/manifests/causal_segments.jsonl" \
  "$ROOT/manifests/export_manifest.jsonl" \
  "$ROOT/manifests/semantic_split_decisions_rescreen_v2.jsonl" \
  "$ROOT/manifests/semantic_rescreen_policy_v2.json" \
  >"$ROOT/reports/manifest_sha256.txt"

df -h /nvmedata >"$ROOT/reports/disk_after_export.txt"
echo "semantic rescreen v2 complete"
