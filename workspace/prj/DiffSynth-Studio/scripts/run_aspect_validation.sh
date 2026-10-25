#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
FORMAL="$PROJECT/validation_results/formal_val_v2"
OUTPUT="$PROJECT/validation_results/formal_aspect_v2"
SUBSET="$PROJECT/data_splits/source_disjoint_v2/validation_subset_15.txt"
TOP_K="${TOP_K:-2}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set explicitly." >&2
  exit 2
fi
mapfile -t SAMPLE_IDS < "$SUBSET"
mapfile -t CANDIDATES < <(
  python3 - "$FORMAL/checkpoint_ranking.json" "$TOP_K" <<'PY'
import json, sys
rows=json.load(open(sys.argv[1], encoding="utf-8"))["ranking"]
for row in rows[:int(sys.argv[2])]:
    print(f"{row['label']}\t{row['checkpoint']}")
PY
)

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate v2v_train
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT"

for candidate in "${CANDIDATES[@]}"; do
  IFS=$'\t' read -r label checkpoint <<< "$candidate"
  python scripts/validate_lora.py \
    --sample-ids "${SAMPLE_IDS[@]}" \
    --lora-path "$checkpoint" \
    --lora-label "$label" \
    --lora-alpha 1.0 \
    --seed 42 \
    --resolution-mode bucket \
    --output-dir "$OUTPUT"
  python scripts/evaluate_lora.py \
    --results-dir "$OUTPUT" \
    --manifest "$OUTPUT/validation_manifest_${label}.json" \
    --output-dir "$OUTPUT/metrics_${label}" \
    --split-name source_disjoint_v2_val_aspect_preserving \
    --variants "$label"
done

python scripts/rank_checkpoints.py --results-root "$OUTPUT"
python scripts/analyze_formal_subgroups.py --results-root "$OUTPUT"
python scripts/build_formal_dashboard.py --results-root "$OUTPUT"
touch "$OUTPUT/ASPECT_VALIDATION_COMPLETE"
