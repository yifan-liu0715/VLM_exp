#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_reference_independent_posenc_scheme4_0729
PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
PYTHON=/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python
CONFIG="$ROOT/configs/train_scheme4_gate_1000.yaml"
PYTHON_PACKAGES="$PROJECT/v2v_formal_0715/environment/python_packages"
MIN_FREE_BYTES=21474836480
GPU="${CUDA_VISIBLE_DEVICES:-}"

if [ -z "$GPU" ]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly one nonzero physical GPU" >&2
  exit 2
fi
if [ "$GPU" = "0" ]; then
  echo "physical GPU0 is forbidden" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "Scheme 4 gate requires exactly one nonzero physical GPU" >&2
  exit 2
fi

FREE_BYTES=$(df -B1 --output=avail /nvmedata | tail -n 1 | tr -d ' ')
if (( FREE_BYTES < MIN_FREE_BYTES )); then
  echo "insufficient /nvmedata space: ${FREE_BYTES} bytes" >&2
  exit 3
fi

export PYTHONPATH="$PYTHON_PACKAGES:$ROOT/scripts:$ROOT/teacher_reference:$PROJECT:${PYTHONPATH:-}"
mkdir -p "$ROOT/logs" "$ROOT/checkpoints"
echo "[launch] Scheme 4 1000-step gate; physical GPU=$GPU; config=$CONFIG"
exec "$PYTHON" "$ROOT/scripts/train_teacher_aligned.py" --config "$CONFIG"
