#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/dataset_h3_capture_semantic_review_2050_coverage355_20260923
SOURCE_ROOT=/mnt/highspeed/users/nina/dataset_h3_capture_causal_full_20260920/sources

python3 "$ROOT/scripts/parallel_export_causal.py" \
  --manifest "$ROOT/manifests/causal_segments.jsonl" \
  --work-root "$ROOT" \
  --source-root "$SOURCE_ROOT" \
  --workers 4 \
  >"$ROOT/logs/export_full_2050.log" 2>&1

python3 "$ROOT/scripts/causal_clip_pipeline.py" validate \
  --manifest "$ROOT/manifests/export_manifest.jsonl" \
  --work-root "$ROOT" \
  --source-root "$SOURCE_ROOT" \
  >"$ROOT/logs/validate_full_2050.log" 2>&1

python3 "$ROOT/scripts/audit_semantic_manifest.py" \
  --manifest "$ROOT/manifests/export_manifest.jsonl" \
  --report "$ROOT/reports/final_manifest_independent_audit.json" \
  >"$ROOT/logs/audit_final_manifest.log" 2>&1

python3 "$ROOT/scripts/audit_semantic_exports.py" \
  --manifest "$ROOT/manifests/export_manifest.jsonl" \
  --work-root "$ROOT" \
  --report "$ROOT/reports/final_export_independent_audit.json" \
  >"$ROOT/logs/audit_final_exports.log" 2>&1

{
  find "$ROOT/clips" -maxdepth 1 -type f -name '*.mp4' -printf '%f %s\n' | sort
  du -sh "$ROOT/clips"
  df -h /nvmedata
} >"$ROOT/reports/final_file_inventory.txt"

sha256sum \
  "$ROOT/manifests/semantic_review_candidates_2050.jsonl" \
  "$ROOT/manifests/causal_segments.jsonl" \
  "$ROOT/manifests/export_manifest.jsonl" \
  >"$ROOT/reports/final_manifest_sha256.txt"
