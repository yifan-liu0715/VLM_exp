#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
OUTPUT="$PROJECT/output_models/wan2.2_lora_quick_v2v_0713"
VALIDATION="$PROJECT/validation_results_0713"
QUEUE_LOG="$VALIDATION/queue_0713.log"
TRAIN_PATTERN='examples/wanvideo/model_training/train.py.*data_0713/v2v_quick_0713/train_targets_0713.csv'
VALIDATION_GPU=1

mkdir -p "$VALIDATION"
exec >> "$QUEUE_LOG" 2>&1
echo "[$(date -Is)] Queue started; waiting for isolated _0713 training. Post-training inference GPU: $VALIDATION_GPU."

while pgrep -f "$TRAIN_PATTERN" >/dev/null; do
  echo "[$(date -Is)] Training still running."
  sleep 60
done

mapfile -t checkpoints < <(find "$OUTPUT" -maxdepth 1 -type f -name 'step-*.safetensors' -print | sort -V)
if (( ${#checkpoints[@]} == 0 )); then
  echo "[$(date -Is)] ERROR: training ended without a step checkpoint."
  exit 1
fi
checkpoint="${checkpoints[-1]}"
echo "[$(date -Is)] Training ended; validating checkpoint: $checkpoint"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate v2v_train
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES="$VALIDATION_GPU" python3 scripts_0713/validate_v2v_lora_0713.py \
  --lora-path "$checkpoint" \
  --sample-ids mymotion__002 mymotion__012 mymotion__015 \
  --seed 42 \
  --denoising-strength 0.45 \
  --num-inference-steps 30 \
  --lora-alpha 1.0

CUDA_VISIBLE_DEVICES="$VALIDATION_GPU" python3 scripts_0713/evaluate_v2v_0713.py
echo "[$(date -Is)] Validation and evaluation completed."
