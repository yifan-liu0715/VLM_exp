#!/usr/bin/env bash
set -uo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808
GPU=${1:?physical GPU id required}
SAMPLE=mymotion__humandance__batch1_mingxing_cropped__002

if [ "$GPU" = 0 ]; then
  echo "GPU0 is forbidden" >&2
  exit 90
fi
if [ "$(df --output=avail /nvmedata | tail -n 1)" -lt $((30 * 1024 * 1024)) ]; then
  echo "less than 30 GiB available on /nvmedata" >&2
  exit 91
fi

cd "$ROOT" || exit 92
rm -f logs/sam2_dance_smoke.exit
{
  echo "physical_gpu=$GPU"
  nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  CUDA_VISIBLE_DEVICES="$GPU" "$ENV/bin/python" scripts/prepare_sam2_masks.py \
    --root "$ROOT" \
    --sample-id "$SAMPLE" \
    --device cuda:0
} > logs/sam2_dance_smoke.log 2>&1
status=$?
printf '%s\n' "$status" > logs/sam2_dance_smoke.exit
exit "$status"
