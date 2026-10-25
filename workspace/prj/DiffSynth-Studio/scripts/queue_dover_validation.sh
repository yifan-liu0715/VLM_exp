#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
FORMAL="$PROJECT/validation_results/formal_val_v2"
DOVER="$PROJECT/third_party/DOVER"
OUTPUT="$FORMAL/dover_mobile"

mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/queue.log") 2>&1
echo "[$(date -Is)] Waiting for formal ranking and DOVER-Mobile weights"
while [[ ! -f "$FORMAL/VALIDATION_QUEUE_COMPLETE" || ! -f "$DOVER/pretrained_weights/DOVER-Mobile.pth.sha256" ]]; do
  sleep 30
done

cd "$PROJECT"
export PYTHONPATH="$DOVER${PYTHONPATH:+:$PYTHONPATH}"
"$PROJECT/.venvs/dover/bin/python" scripts/evaluate_dover.py \
  --results-root "$FORMAL" \
  --dover-root "$DOVER" \
  --output-dir "$OUTPUT" \
  --top-k 3 \
  --device cpu
touch "$OUTPUT/DOVER_COMPLETE"
echo "[$(date -Is)] DOVER-Mobile evaluation completed"
