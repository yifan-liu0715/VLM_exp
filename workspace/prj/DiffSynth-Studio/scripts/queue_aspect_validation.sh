#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
MULTISEED="$PROJECT/validation_results/formal_multiseed"
OUTPUT="$PROJECT/validation_results/formal_aspect_v2"
GPU=5

mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/queue.log") 2>&1
echo "[$(date -Is)] Waiting for multi-seed validation"
while [[ ! -f "$MULTISEED/MULTISEED_COMPLETE" ]]; do
  sleep 30
done
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  if [[ "$used" -lt 1024 && "$util" -lt 10 ]]; then
    break
  fi
  echo "[$(date -Is)] GPU $GPU busy: used=${used}MiB util=${util}%"
  sleep 30
done
echo "[$(date -Is)] Starting aspect-preserving top-checkpoint validation"
CUDA_VISIBLE_DEVICES="$GPU" "$PROJECT/scripts/run_aspect_validation.sh"
echo "[$(date -Is)] Aspect-preserving validation completed"
