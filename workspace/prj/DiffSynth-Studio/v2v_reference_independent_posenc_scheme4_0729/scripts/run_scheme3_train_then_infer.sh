#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_native_resolution_scheme3_0727
PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
PYTHON=/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python
PYTHON_PACKAGES="$PROJECT/v2v_formal_0715/environment/python_packages"
GPU="${1:?pass one nonzero physical GPU id}"
ADAPTER="$ROOT/recommended/final_adapter/v2v_lora_and_source_patch.safetensors"

if [[ ! "$GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU0 is forbidden and exactly one physical GPU is required" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PYTHON_PACKAGES:$ROOT/scripts:$ROOT/teacher_reference:$PROJECT:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs" "$ROOT/validation/logs"
exec > >(tee -a "$ROOT/logs/controller.log") 2>&1

echo "$$" > "$ROOT/logs/controller.pid"
echo "[controller] $(date -Is) validating Scheme 3 on physical GPU $GPU"
cd "$ROOT"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" "$ROOT/scripts/preflight_scheme3_dataset.py"

echo "[controller] $(date -Is) starting native-resolution training"
bash "$ROOT/scripts/launch_scheme3_train.sh" 2>&1 | tee "$ROOT/logs/train.log"
if [ ! -s "$ADAPTER" ]; then
  echo "final adapter missing after training: $ADAPTER" >&2
  exit 4
fi
echo "[controller] $(date -Is) training complete; starting held-out inference"

SAMPLES=(
  "mymotion__051"
  "mymotion__single_talking_face__datatrain__002"
  "mymotion__humandance__batch1_mingxing_cropped__002"
)

for sample_id in "${SAMPLES[@]}"; do
  output_dir="$ROOT/validation/$sample_id"
  log_file="$ROOT/validation/logs/${sample_id}.log"
  mkdir -p "$output_dir"
  echo "[controller] $(date -Is) inference sample=$sample_id"
  "$PYTHON" "$ROOT/scripts/infer_scheme3_native.py" \
    --config "$ROOT/configs/train_scheme3_native_2000.yaml" \
    --manifest "$ROOT/metadata/test_manifest.csv" \
    --sample-id "$sample_id" \
    --adapter "$ADAPTER" \
    --output-dir "$output_dir" \
    2>&1 | tee "$log_file"
done

echo "[controller] $(date -Is) Scheme 3 training and inference complete"
touch "$ROOT/validation/COMPLETE"
