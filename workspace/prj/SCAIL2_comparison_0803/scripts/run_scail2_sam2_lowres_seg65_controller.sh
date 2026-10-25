#!/usr/bin/env bash
set -uo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
PYTHON=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python
GPU=${1:?physical GPU id required}
SAMPLE=mymotion__humandance__batch1_mingxing_cropped__002
PLAN=$ROOT/manifests/inference_plan_sam2_lowres_seg65.json

if [ "$GPU" = 0 ]; then
  echo "GPU0 is forbidden" >&2
  exit 90
fi

cd "$ROOT" || exit 91
export PYTORCH_ALLOC_CONF=expandable_segments:True
"$PYTHON" scripts/run_scail2_sam2_gate.py \
  --root "$ROOT" \
  --plan "$PLAN" \
  --sample-id "$SAMPLE" \
  --physical-gpu "$GPU" \
  > logs/scail2_sam2_lowres_seg65_controller.log 2>&1
status=$?
printf '%s\n' "$status" > logs/scail2_sam2_lowres_seg65_controller.exit
exit "$status"
