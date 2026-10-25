#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
SUBSET="$PROJECT/data_splits/source_disjoint_v2/validation_subset_15.txt"
OUTPUT="$PROJECT/validation_results/formal_val_v2"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=N $0 CHECKPOINT LABEL [ALPHA]" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set explicitly." >&2
  exit 2
fi

CHECKPOINT=$(realpath "$1")
LABEL="$2"
ALPHA="${3:-1.0}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi
mapfile -t SAMPLE_IDS < "$SUBSET"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate v2v_train
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python scripts/validate_lora.py \
  --sample-ids "${SAMPLE_IDS[@]}" \
  --lora-path "$CHECKPOINT" \
  --lora-label "$LABEL" \
  --lora-alpha "$ALPHA" \
  --output-dir "$OUTPUT"

python scripts/evaluate_lora.py \
  --results-dir "$OUTPUT" \
  --manifest "$OUTPUT/validation_manifest_${LABEL}.json" \
  --output-dir "$OUTPUT/metrics_${LABEL}" \
  --split-name source_disjoint_v2_val \
  --variants "$LABEL"
