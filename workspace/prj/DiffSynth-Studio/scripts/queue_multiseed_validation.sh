#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
FORMAL="$PROJECT/validation_results/formal_val_v2"
OUTPUT="$PROJECT/validation_results/formal_multiseed"
GPU=5

mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/queue.log") 2>&1

echo "[$(date -Is)] Waiting for formal validation"
while [[ ! -f "$FORMAL/VALIDATION_QUEUE_COMPLETE" ]]; do
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

echo "[$(date -Is)] Starting top-checkpoint multi-seed validation"
CUDA_VISIBLE_DEVICES="$GPU" "$PROJECT/scripts/run_multiseed_validation.sh"
echo "[$(date -Is)] Multi-seed validation completed"
