#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_cfgfix_diagnostic_0720
PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
FORMAL_ROOT="${PROJECT}/v2v_formal_0715"
PYTHON_BIN=/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python
PID_FILE="${ROOT}/cfg_diagnostic.pid"
LOG_FILE="${ROOT}/logs/infer_recommended_cfg5.log"
TMP_ROOT="${ROOT}/tmp"

mkdir -p "${TMP_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "v2v_train python not found: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if kill -0 "${existing_pid}" 2>/dev/null; then
    echo "CFG diagnostic already running as PID ${existing_pid}"
    exit 0
  fi
fi

PHYSICAL_GPU="${PHYSICAL_GPU:-2}"
if [[ ! "${PHYSICAL_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PHYSICAL_GPU must be one nonzero GPU index" >&2
  exit 2
fi

nohup env \
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
  PYTHONPATH="${ROOT}/scripts:${FORMAL_ROOT}/environment/python_packages:${FORMAL_ROOT}/scripts:${PROJECT}" \
  TMPDIR="${TMP_ROOT}" \
  TEMP="${TMP_ROOT}" \
  TMP="${TMP_ROOT}" \
  "${PYTHON_BIN}" "${ROOT}/scripts/infer_v2v_comparison_cfg.py" \
  --sample-id mymotion__002 \
  --manifest "${PROJECT}/v2v_corrected_tiers_0719/metadata/metadata_corrected_direction_0719.csv" \
  --config "${PROJECT}/v2v_corrected_tiers_0719/resolved_train_config.yaml" \
  --adapter "${PROJECT}/v2v_corrected_tiers_0719/recommended/final_adapter/v2v_lora_and_source_patch.safetensors" \
  --output-dir "${ROOT}/validation_recommended_cfg5" \
  --num-frames 33 \
  --steps 30 \
  --seed 42 \
  --guidance-scale 5 \
  >"${LOG_FILE}" 2>&1 &

pid=$!
echo "${pid}" >"${PID_FILE}"
echo "started PID ${pid}; log=${LOG_FILE}; physical GPU=${PHYSICAL_GPU}"
