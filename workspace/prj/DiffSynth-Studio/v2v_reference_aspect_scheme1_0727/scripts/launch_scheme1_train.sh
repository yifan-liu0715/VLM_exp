#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_reference_aspect_scheme1_0727
PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
PYTHON=/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python
CONFIG="$ROOT/configs/train_scheme1_2000.yaml"
PYTHON_PACKAGES="$PROJECT/v2v_formal_0715/environment/python_packages"
# The measured compact footprint is about 1.5 GiB because recovery and latest
# adapter files overwrite in place. Keep a 5 GiB hard floor on this diagnostic.
MIN_FREE_KB=$((5 * 1024 * 1024))

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly one nonzero physical GPU" >&2
  exit 2
fi
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Scheme 1 requires exactly one nonzero physical GPU" >&2
  exit 2
fi

FREE_KB=$(df -Pk /nvmedata | awk 'NR==2 {print $4}')
if (( FREE_KB < MIN_FREE_KB )); then
  echo "insufficient /nvmedata space: ${FREE_KB} KiB" >&2
  exit 3
fi

export PYTHONPATH="$PYTHON_PACKAGES:$ROOT/scripts:$ROOT/teacher_reference:$PROJECT:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs" "$ROOT/checkpoints"
echo "[launch] Scheme 1; physical GPU=$CUDA_VISIBLE_DEVICES; GPU0 forbidden; config=$CONFIG"
exec "$PYTHON" "$ROOT/scripts/train_teacher_aligned.py" --config "$CONFIG"
