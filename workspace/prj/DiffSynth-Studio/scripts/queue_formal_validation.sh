#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
TRAIN_SESSION=nina_wan22_formal_v2_20260712
MODEL_DIR="$PROJECT/output_models/wan2.2_lora_formal_v2_r16_lr5e5"
RESULTS="$PROJECT/validation_results/formal_val_v2"
GPU=5

mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/queue.log") 2>&1

echo "[$(date -Is)] Waiting for training session: $TRAIN_SESSION"
while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 30
done
echo "[$(date -Is)] Training session ended; waiting for GPU $GPU to be idle"

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  if [[ "$used" -lt 1024 && "$util" -lt 10 ]]; then
    break
  fi
  echo "[$(date -Is)] GPU $GPU still busy: used=${used}MiB util=${util}%"
  sleep 30
done

mapfile -t CHECKPOINTS < <(find "$MODEL_DIR" -maxdepth 1 -type f -name 'step-*.safetensors' | sort -V)
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No checkpoints found in $MODEL_DIR" >&2
  exit 1
fi

echo "[$(date -Is)] Validating ${#CHECKPOINTS[@]} checkpoints on GPU $GPU"
for checkpoint in "${CHECKPOINTS[@]}"; do
  stem=$(basename "$checkpoint" .safetensors)
  label="formal_v2_${stem}"
  echo "[$(date -Is)] Begin $label"
  CUDA_VISIBLE_DEVICES="$GPU" "$PROJECT/scripts/run_checkpoint_validation.sh" \
    "$checkpoint" "$label" 1.0
  echo "[$(date -Is)] Finished $label"
done

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate v2v_train
python "$PROJECT/scripts/rank_checkpoints.py" --results-root "$RESULTS"
python "$PROJECT/scripts/prepare_blind_review.py" --results-root "$RESULTS"
python "$PROJECT/scripts/build_formal_dashboard.py" --results-root "$RESULTS"

touch "$RESULTS/VALIDATION_QUEUE_COMPLETE"
echo "[$(date -Is)] All checkpoint validations completed"
