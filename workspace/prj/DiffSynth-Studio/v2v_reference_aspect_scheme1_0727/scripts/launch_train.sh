#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_teacher_aligned_recommended_0724
PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
PYTHON=/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python
CONFIG="$ROOT/configs/train_teacher_aligned_20000.yaml"
PYTHON_PACKAGES="$PROJECT/v2v_formal_0715/environment/python_packages"
MIN_FREE_KB=$((20 * 1024 * 1024))

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must list one to three nonzero physical GPUs" >&2
  exit 2
fi
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#GPUS[@]} > 3 )); then
  echo "at most three physical GPUs are allowed" >&2
  exit 2
fi
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU identifier: $gpu" >&2
    exit 2
  fi
  if [[ "$gpu" == "0" ]]; then
    echo "GPU0 is forbidden" >&2
    exit 2
  fi
  if [[ -n "${SEEN[$gpu]:-}" ]]; then
    echo "duplicate GPU identifier: $gpu" >&2
    exit 2
  fi
  SEEN[$gpu]=1
done

FREE_KB=$(df -Pk /nvmedata | awk 'NR==2 {print $4}')
if (( FREE_KB < MIN_FREE_KB )); then
  echo "insufficient /nvmedata space: ${FREE_KB} KiB" >&2
  exit 3
fi

export PYTHONPATH="$PYTHON_PACKAGES:$ROOT/scripts:$ROOT/teacher_reference:$PROJECT:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs"
echo "[launch] physical GPUs=$CUDA_VISIBLE_DEVICES; GPU0 forbidden; config=$CONFIG"
exec "$PYTHON" "$ROOT/scripts/train_teacher_aligned.py" --config "$CONFIG"
