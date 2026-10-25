#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
FORMAL="$PROJECT/validation_results/formal_val_v2"
OUTPUT_ROOT="$PROJECT/validation_results/formal_multiseed"
SUBSET="$PROJECT/data_splits/source_disjoint_v2/validation_subset_15.txt"
TOP_K="${TOP_K:-2}"
SEEDS=(123 20260712)

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set explicitly." >&2
  exit 2
fi
if [[ ! -f "$FORMAL/checkpoint_ranking.json" ]]; then
  echo "Formal checkpoint ranking is not ready." >&2
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

python scripts/analyze_formal_subgroups.py --results-root "$FORMAL"
python scripts/build_formal_dashboard.py --results-root "$FORMAL"

for seed in "${SEEDS[@]}"; do
  output="$OUTPUT_ROOT/seed-$seed"
  mkdir -p "$output"
  for candidate in "${CANDIDATES[@]}"; do
    IFS=$'\t' read -r label checkpoint <<< "$candidate"
    python scripts/validate_lora.py \
      --sample-ids "${SAMPLE_IDS[@]}" \
      --lora-path "$checkpoint" \
      --lora-label "$label" \
      --lora-alpha 1.0 \
      --seed "$seed" \
      --output-dir "$output"
    python scripts/evaluate_lora.py \
      --results-dir "$output" \
      --manifest "$output/validation_manifest_${label}.json" \
      --output-dir "$output/metrics_${label}" \
      --split-name source_disjoint_v2_val_seed_${seed} \
      --variants "$label"
  done
done

python scripts/aggregate_multiseed.py \
  --formal-root "$FORMAL" \
  --multiseed-root "$OUTPUT_ROOT" \
  --top-k "$TOP_K"

touch "$OUTPUT_ROOT/MULTISEED_COMPLETE"
