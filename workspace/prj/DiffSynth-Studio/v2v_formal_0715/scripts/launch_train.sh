#!/usr/bin/env bash
set -euo pipefail

ROOT="/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio"
FORMAL_ROOT="${ROOT}/v2v_formal_0715"
PYTHON="/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python"
CONFIG="${FORMAL_ROOT}/configs/train_v2v_lora_0715.yaml"
GPU_IDS="${GPU_IDS:-1,2,3}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} == 0 || ${#GPUS[@]} > 3 )); then
  echo "Expected one to three GPUs, got: ${GPU_IDS}" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: ${gpu}" >&2
    exit 2
  fi
  if [[ "${gpu}" == "0" ]]; then
    echo "GPU0 is forbidden for this run" >&2
    exit 2
  fi
  free_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( free_mib < 70000 )); then
    echo "GPU${gpu} is not sufficiently free (${free_mib} MiB); aborting without starting" >&2
    exit 3
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${FORMAL_ROOT}/environment/python_packages:${FORMAL_ROOT}/scripts:${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

COMMAND="${PYTHON} ${FORMAL_ROOT}/scripts/train_v2v_lora.py --config ${CONFIG}"
echo "$$" > "${FORMAL_ROOT}/reports/formal_training.pid"
"${PYTHON}" "${FORMAL_ROOT}/scripts/write_run_manifest.py" \
  --config "${CONFIG}" \
  --gpus "${GPU_IDS}" \
  --command "${COMMAND}"

exec "${PYTHON}" "${FORMAL_ROOT}/scripts/train_v2v_lora.py" --config "${CONFIG}"
