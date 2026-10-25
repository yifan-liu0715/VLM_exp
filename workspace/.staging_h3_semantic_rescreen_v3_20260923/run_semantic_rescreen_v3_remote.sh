#!/usr/bin/env bash
set -euo pipefail

STAGE=${STAGE:-/nvmedata/workspace2/users/nina/.staging_h3_semantic_rescreen_v3_20260923}
V2=/nvmedata/workspace2/users/nina/dataset_h3_capture_semantic_rescreen_v2_2045_coverage357_20260923
V3=/nvmedata/workspace2/users/nina/dataset_h3_capture_semantic_rescreen_v3_2016_coverage357_20260923
OLD=/nvmedata/workspace2/users/nina/dataset_h3_capture_semantic_review_2050_coverage355_20260923
SOURCE_ROOT=/mnt/highspeed/users/nina/dataset_h3_capture_causal_full_20260920/sources
CANDIDATE_SHA=7f2650af237fdabcefcaa5fd6b124636eaf3d1336066f42d61ee79d0f9bba4e0
REMOVED_SHA=38e509299e20f5b8cdc6fe98cf392cf59050c9d5ab319c0e0c8c55ff1451aead
POLICY_SHA=8dcd9ddc236e6c2c42305e048bf28e2d0aa6161d49aab4dc096cbe500cb82015

echo "[$(date -Is)] phase=materialize start"
python3 "$STAGE/materialize_semantic_rescreen_v3.py" \
  --v2-root "$V2" \
  --v3-root "$V3" \
  --candidates "$STAGE/semantic_review_candidates_2016.jsonl" \
  --removed "$STAGE/continuous_long_event_removed_29.jsonl" \
  --policy "$STAGE/semantic_continuous_long_event_policy_v3.json" \
  --expected-candidate-sha256 "$CANDIDATE_SHA" \
  --expected-removed-sha256 "$REMOVED_SHA" \
  --expected-policy-sha256 "$POLICY_SHA" \
  >"$STAGE/materialize.log" 2>&1

cp -p "$STAGE/materialize.log" "$V3/logs/materialize.log"
cp -p "$STAGE/materialize_semantic_rescreen_v3.py" "$V3/scripts/"
cp -p "$STAGE/filter_continuous_long_events.py" "$V3/scripts/"
cp -p "$STAGE/causal_clip_pipeline.py" "$V3/scripts/"
cp -p "$STAGE/audit_semantic_manifest.py" "$V3/scripts/"
cp -p "$STAGE/audit_semantic_exports.py" "$V3/scripts/"
cp -p "$STAGE/run_semantic_rescreen_v3_remote.sh" "$V3/scripts/"
echo "[$(date -Is)] phase=materialize complete"

echo "[$(date -Is)] phase=manifest_audit start"
python3 "$V3/scripts/audit_semantic_manifest.py" \
  --manifest "$V3/manifests/semantic_review_candidates_2016.jsonl" \
  --report "$V3/reports/independent_manifest_audit.json" \
  >"$V3/logs/audit_candidates.log" 2>&1
python3 - "$V3/reports/independent_manifest_audit.json" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["audit"]
expected = {
    "row_count": 2016,
    "unique_id_count": 2016,
    "empty_id_count": 0,
    "source_coverage_count": 357,
    "unique_export_window_count": 2016,
    "duplicate_export_window_group_count": 0,
    "duplicate_core_group_count": 0,
    "invalid_source_sha256_count": 0,
    "contract_error_count": 0,
    "pending_user_review_false_count": 0,
    "semantic_feedback_verified_nonfalse_count": 0,
    "navigation_only_count": 0,
    "export_ineligible_count": 0,
}
errors = {key: {"expected": value, "actual": audit.get(key)} for key, value in expected.items() if audit.get(key) != value}
if errors:
    raise SystemExit(f"manifest audit failed: {errors}")
print(json.dumps(expected, sort_keys=True))
PY
echo "[$(date -Is)] phase=manifest_audit complete"

echo "[$(date -Is)] phase=full_decode_validation start"
python3 "$V3/scripts/causal_clip_pipeline.py" validate \
  --manifest "$V3/manifests/export_manifest.jsonl" \
  --work-root "$V3" \
  --source-root "$SOURCE_ROOT" \
  --skip-source-sha \
  >"$V3/logs/validate_full_decode.log" 2>&1
echo "[$(date -Is)] phase=full_decode_validation complete"

echo "[$(date -Is)] phase=independent_output_audit start"
python3 "$V3/scripts/audit_semantic_exports.py" \
  --manifest "$V3/manifests/export_manifest.jsonl" \
  --work-root "$V3" \
  --report "$V3/reports/final_export_independent_audit.json" \
  >"$V3/logs/audit_final_exports.log" 2>&1

python3 - "$OLD" "$V2" "$V3" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

old, v2, v3 = map(Path, sys.argv[1:])

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

old_rows = load_jsonl(old / "manifests/semantic_review_candidates_2050.jsonl")
v2_rows = load_jsonl(v2 / "manifests/semantic_review_candidates_2045.jsonl")
v3_rows = load_jsonl(v3 / "manifests/semantic_review_candidates_2016.jsonl")
removed_rows = load_jsonl(v3 / "manifests/continuous_long_event_removed_29.jsonl")
exports = load_jsonl(v3 / "manifests/export_manifest.jsonl")
old_ids = {row["candidate_id"] for row in old_rows}
v2_ids = {row["candidate_id"] for row in v2_rows}
v3_ids = {row["candidate_id"] for row in v3_rows}
removed_ids = {row["candidate_id"] for row in removed_rows}
export_ids = {row["clip_id"] for row in exports}
clip_paths = sorted((v3 / "clips").glob("*.mp4"))
sidecar_paths = sorted((v3 / "state/exports").glob("*.json"))
review_paths = sorted((v3 / "review_additions_14").glob("*.mp4"))
keypoint_paths = sorted((v3 / "hand_keypoints").rglob("*_hands.json")) if (v3 / "hand_keypoints").is_dir() else []
material = load_json(v3 / "reports/materialization_report.json")
manifest_audit = load_json(v3 / "reports/independent_manifest_audit.json")["audit"]
validation = load_json(v3 / "reports/validation.json")
export_audit = load_json(v3 / "reports/final_export_independent_audit.json")
v2_exports = {row["clip_id"]: row for row in load_jsonl(v2 / "manifests/export_manifest.jsonl")}
hardlink_matches = 0
independent_sidecars = 0
for row in exports:
    clip_id = row["clip_id"]
    old_video = Path(v2_exports[clip_id]["output_video"])
    new_video = Path(row["output_video"])
    old_stat = old_video.stat()
    new_stat = new_video.stat()
    hardlink_matches += (old_stat.st_dev, old_stat.st_ino) == (new_stat.st_dev, new_stat.st_ino)
    old_sidecar = v2 / "state/exports" / f"{clip_id}.json"
    new_sidecar = v3 / "state/exports" / f"{clip_id}.json"
    independent_sidecars += (old_sidecar.stat().st_dev, old_sidecar.stat().st_ino) != (new_sidecar.stat().st_dev, new_sidecar.stat().st_ino)

report = {
    "schema_version": 1,
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "ready_for_manual_review",
    "v3_root": str(v3),
    "candidate_count": len(v3_rows),
    "export_count": len(exports),
    "clip_file_count": len(clip_paths),
    "sidecar_file_count": len(sidecar_paths),
    "source_coverage_count": len({row["source_sha256"] for row in v3_rows}),
    "total_duration_sec": round(sum(float(row["duration_sec"]) for row in exports), 6),
    "total_output_size_bytes": sum(path.stat().st_size for path in clip_paths),
    "unique_output_sha256_count": len({row["output_sha256"] for row in exports}),
    "hardlink_to_v2_count": hardlink_matches,
    "independent_sidecar_count": independent_sidecars,
    "review_additions_count": len(review_paths),
    "hand_keypoint_file_count": len(keypoint_paths),
    "old_2050_added_count": len(v3_ids - old_ids),
    "old_2050_removed_count": len(old_ids - v3_ids),
    "old_2050_common_count": len(old_ids & v3_ids),
    "v2_added_count": len(v3_ids - v2_ids),
    "v2_removed_count": len(v2_ids - v3_ids),
    "v2_common_count": len(v2_ids & v3_ids),
    "removed_partition_count": len(removed_ids),
    "manual_presence_override_reason_count": sum(bool(row.get("manual_presence_override_reason")) for row in v3_rows),
    "pending_user_review_count": sum(row.get("pending_user_review") is True for row in v3_rows),
    "semantic_feedback_verified_nonfalse_count": sum(row.get("semantic_feedback_verified") is not False for row in v3_rows),
    "candidate_manifest_sha256": sha256_file(v3 / "manifests/semantic_review_candidates_2016.jsonl"),
    "export_manifest_sha256": sha256_file(v3 / "manifests/export_manifest.jsonl"),
    "hand_keypoint_run": False,
    "training_run": False,
}
expected = {
    "candidate_count": 2016,
    "export_count": 2016,
    "clip_file_count": 2016,
    "sidecar_file_count": 2016,
    "source_coverage_count": 357,
    "unique_output_sha256_count": 2016,
    "hardlink_to_v2_count": 2016,
    "independent_sidecar_count": 2016,
    "review_additions_count": 14,
    "hand_keypoint_file_count": 0,
    "old_2050_added_count": 14,
    "old_2050_removed_count": 48,
    "old_2050_common_count": 2002,
    "v2_added_count": 0,
    "v2_removed_count": 29,
    "v2_common_count": 2016,
    "removed_partition_count": 29,
    "manual_presence_override_reason_count": 14,
    "pending_user_review_count": 2016,
    "semantic_feedback_verified_nonfalse_count": 0,
}
errors = {key: {"expected": value, "actual": report.get(key)} for key, value in expected.items() if report.get(key) != value}
if v3_ids | removed_ids != v2_ids or v3_ids & removed_ids:
    errors["v2_partition"] = {"expected": "exact disjoint partition", "actual": "mismatch"}
if export_ids != v3_ids:
    errors["export_id_set"] = {"expected": len(v3_ids), "actual": len(export_ids)}
if material.get("materialized_count") != 2016 or material.get("removed_count") != 29:
    errors["materialization_report"] = material
if manifest_audit.get("contract_error_count") or manifest_audit.get("duplicate_export_window_group_count") or manifest_audit.get("duplicate_core_group_count"):
    errors["manifest_audit"] = manifest_audit
if validation.get("manifest_count") != 2016 or validation.get("valid_count") != 2016 or validation.get("error_count") != 0:
    errors["full_decode_validation"] = validation
if export_audit.get("manifest_count") != 2016 or export_audit.get("unique_clip_id_count") != 2016 or export_audit.get("unique_output_path_count") != 2016:
    errors["export_audit_counts"] = export_audit
for field in ("export_error_count", "missing_output_count", "unexpected_clip_file_count", "output_sha256_mismatch_count", "duplicate_output_sha256_group_count", "semantic_feedback_verified_nonfalse_count", "hand_keypoint_file_count"):
    if export_audit.get(field) != 0:
        errors[f"export_audit_{field}"] = export_audit.get(field)
report["errors"] = errors
rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
temporary = v3 / "reports/.final_dataset_audit.json.tmp"
temporary.write_text(rendered, encoding="utf-8")
os.replace(temporary, v3 / "reports/final_dataset_audit.json")
print(json.dumps(report, ensure_ascii=False))
if errors:
    raise SystemExit(2)
PY

sha256sum \
  "$V3/manifests/semantic_review_candidates_2016.jsonl" \
  "$V3/manifests/continuous_long_event_removed_29.jsonl" \
  "$V3/manifests/semantic_continuous_long_event_policy_v3.json" \
  "$V3/manifests/causal_segments.jsonl" \
  "$V3/manifests/export_manifest.jsonl" \
  >"$V3/reports/manifest_sha256.txt"

printf '%s\n' "$V3" >"$V2/SUPERSEDED_BY"
printf '%s\n' 'ready_for_manual_review' >"$V3/READY_FOR_MANUAL_REVIEW"
df -h /nvmedata >"$V3/reports/disk_after_materialization.txt"
echo "[$(date -Is)] phase=independent_output_audit complete"
echo "[$(date -Is)] status=complete root=$V3"
