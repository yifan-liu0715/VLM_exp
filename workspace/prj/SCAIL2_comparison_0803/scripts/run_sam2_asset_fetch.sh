#!/usr/bin/env bash
set -uo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
PYTHON=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python
cd "$ROOT" || exit 90
rm -f logs/sam2_asset_fetch.exit
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/fetch_sam2_fallback_assets.py --root "$ROOT" \
  > logs/sam2_asset_fetch.log 2>&1
status=$?
printf '%s\n' "$status" > logs/sam2_asset_fetch.exit
exit "$status"
