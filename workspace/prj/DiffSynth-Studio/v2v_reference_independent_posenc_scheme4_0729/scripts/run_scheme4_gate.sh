#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_reference_independent_posenc_scheme4_0729
PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
PYTHON=/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python
PYTHON_PACKAGES="$PROJECT/v2v_formal_0715/environment/python_packages"
GPU="${1:?pass one nonzero physical GPU id}"
SCHEME3_ADAPTER="$PROJECT/v2v_native_resolution_scheme3_0727/recommended/final_adapter/v2v_lora_and_source_patch.safetensors"
SCHEME4_ADAPTER="$ROOT/recommended/final_adapter/v2v_lora_source_reference_posenc.safetensors"
MIN_FREE_BYTES=21474836480

if [ "$GPU" = "0" ]; then
  echo "physical GPU0 is forbidden" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU0 is forbidden and exactly one physical GPU is required" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PYTHON_PACKAGES:$ROOT/scripts:$ROOT/teacher_reference:$PROJECT:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs" "$ROOT/validation/logs"
exec > >(tee -a "$ROOT/logs/controller.log") 2>&1

echo "$$" > "$ROOT/logs/controller.pid"
FREE_BYTES=$(df -B1 --output=avail /nvmedata | tail -n 1 | tr -d ' ')
if (( FREE_BYTES < MIN_FREE_BYTES )); then
  echo "insufficient /nvmedata space: ${FREE_BYTES} bytes" >&2
  exit 3
fi

echo "[controller] $(date -Is) validating Scheme 4 on physical GPU $GPU"
cd "$ROOT"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" "$ROOT/scripts/preflight_scheme4.py" \
  --config "$ROOT/configs/train_scheme4_gate_1000.yaml" \
  --output "$ROOT/validation/cpu_preflight_report.json"

echo "[controller] $(date -Is) starting Scheme 4 gate training"
bash "$ROOT/scripts/launch_scheme4_train.sh" 2>&1 | tee "$ROOT/logs/train.log"
if [ ! -s "$SCHEME4_ADAPTER" ]; then
  echo "final adapter missing after training: $SCHEME4_ADAPTER" >&2
  exit 4
fi
if [ ! -s "$SCHEME3_ADAPTER" ]; then
  echo "Scheme 3 comparison adapter missing: $SCHEME3_ADAPTER" >&2
  exit 4
fi
echo "[controller] $(date -Is) training complete; starting held-out inference"

SAMPLES=(
  "mymotion__051"
  "mymotion__single_talking_face__datatrain__002"
  "mymotion__humandance__batch1_mingxing_cropped__002"
)

for sample_id in "${SAMPLES[@]}"; do
  output_dir="$ROOT/validation/gate_1000"
  log_file="$ROOT/validation/logs/${sample_id}.log"
  mkdir -p "$output_dir"
  echo "[controller] $(date -Is) inference sample=$sample_id"
  "$PYTHON" "$ROOT/scripts/infer_scheme4.py" \
    --config "$ROOT/configs/train_scheme4_gate_1000.yaml" \
    --manifest "$ROOT/metadata/test_manifest.csv" \
    --sample-id "$sample_id" \
    --scheme3-adapter "$SCHEME3_ADAPTER" \
    --scheme4-adapter "$SCHEME4_ADAPTER" \
    --output-dir "$output_dir" \
    2>&1 | tee "$log_file"
done

echo "[controller] $(date -Is) Scheme 4 gate and inference complete"
touch "$ROOT/validation/COMPLETE"
