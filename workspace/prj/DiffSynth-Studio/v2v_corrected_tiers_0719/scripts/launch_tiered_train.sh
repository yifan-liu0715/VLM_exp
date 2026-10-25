#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio"
RUN_ROOT="${PROJECT_ROOT}/v2v_corrected_tiers_0719"
FORMAL_ROOT="${PROJECT_ROOT}/v2v_formal_0715"
PYTHON="/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python"
CONFIG="${RUN_ROOT}/configs/train_v2v_corrected_tiers_0719.yaml"
GPU_IDS="${GPU_IDS:-1,2,4}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} == 0 || ${#GPUS[@]} > 3 )); then
  echo "Expected one to three GPUs, got ${GPU_IDS}" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ || "${gpu}" == "0" ]]; then
    echo "Invalid or forbidden GPU id: ${gpu}" >&2
    exit 2
  fi
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( free_mib < 70000 )); then
    echo "GPU${gpu} is not sufficiently free (${free_mib} MiB); aborting" >&2
    exit 3
  fi
done

free_kib="$(df --output=avail "${RUN_ROOT}" | tail -1 | tr -d ' ')"
if (( free_kib < 31457280 )); then
  echo "Less than 30 GiB free on the run filesystem; aborting" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${RUN_ROOT}/scripts:${FORMAL_ROOT}/environment/python_packages:${FORMAL_ROOT}/scripts:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[launch] physical GPUs=${GPU_IDS}; GPU0 forbidden; config=${CONFIG}"
exec "${PYTHON}" "${RUN_ROOT}/scripts/train_v2v_lora_tiered.py" --config "${CONFIG}" "$@"
