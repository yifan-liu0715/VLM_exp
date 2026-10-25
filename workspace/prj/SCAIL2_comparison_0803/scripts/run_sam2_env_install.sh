#!/usr/bin/env bash
set -uo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
cd "$ROOT" || exit 90
rm -f logs/sam2_env_install.exit
CUDA_VISIBLE_DEVICES='' bash scripts/create_sam2_mask_env.sh > logs/sam2_env_install.log 2>&1
status=$?
printf '%s\n' "$status" > logs/sam2_env_install.exit
exit "$status"
