#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/dataset_h3_capture_semantic_review_2050_coverage355_20260923
SOURCE_ROOT=/mnt/highspeed/users/nina/dataset_h3_capture_causal_full_20260920/sources

python3 "$ROOT/scripts/build_semantic_review_candidates.py" \
  --inventory "$ROOT/manifests/source_inventory_portable.jsonl" \
  --presence "$ROOT/manifests/hand_presence_proposals.jsonl" \
  --v3 "$ROOT/manifests/proposals_v3_all.jsonl" \
  --high-recall "$ROOT/manifests/causal_proposals_high_recall.jsonl" \
  --decisions "$ROOT/manifests/semantic_split_decisions.jsonl" \
  --work-root "$ROOT" \
  --target-count 2050 \
  >"$ROOT/logs/build_candidates.log" 2>&1

python3 "$ROOT/scripts/audit_semantic_manifest.py" \
  --manifest "$ROOT/manifests/semantic_review_candidates_2050.jsonl" \
  --report "$ROOT/reports/semantic_candidates_independent_audit.json" \
  >"$ROOT/logs/audit_candidates.log" 2>&1

python3 "$ROOT/scripts/causal_clip_pipeline.py" prepare \
  --proposals "$ROOT/manifests/semantic_review_candidates_2050.jsonl" \
  --work-root "$ROOT" \
  --source-root "$SOURCE_ROOT" \
  --max-duration 20 \
  >"$ROOT/logs/prepare.log" 2>&1

python3 "$ROOT/scripts/audit_semantic_manifest.py" \
  --manifest "$ROOT/manifests/causal_segments.jsonl" \
  --ids "$ROOT/manifests/semantic_smoke_ids_10.txt" \
  --output "$ROOT/smoke_10/manifests/causal_segments.jsonl" \
  --report "$ROOT/reports/prepared_manifest_independent_audit.json" \
  >"$ROOT/logs/audit_prepared.log" 2>&1

python3 "$ROOT/scripts/causal_clip_pipeline.py" export \
  --manifest "$ROOT/smoke_10/manifests/causal_segments.jsonl" \
  --work-root "$ROOT/smoke_10" \
  --source-root "$SOURCE_ROOT" \
  --smoke-limit 10 \
  >"$ROOT/logs/export_smoke_10.log" 2>&1

python3 "$ROOT/scripts/causal_clip_pipeline.py" validate \
  --manifest "$ROOT/smoke_10/manifests/export_manifest.jsonl" \
  --work-root "$ROOT/smoke_10" \
  --source-root "$SOURCE_ROOT" \
  >"$ROOT/logs/validate_smoke_10.log" 2>&1

sha256sum \
  "$ROOT/manifests/semantic_review_candidates_2050.jsonl" \
  "$ROOT/manifests/causal_segments.jsonl" \
  "$ROOT/smoke_10/manifests/causal_segments.jsonl" \
  "$ROOT/smoke_10/manifests/export_manifest.jsonl" \
  >"$ROOT/reports/manifest_sha256.txt"
