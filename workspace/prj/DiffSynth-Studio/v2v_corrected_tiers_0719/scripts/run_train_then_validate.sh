#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio"
RUN_ROOT="${PROJECT_ROOT}/v2v_corrected_tiers_0719"
FORMAL_ROOT="${PROJECT_ROOT}/v2v_formal_0715"
PYTHON="/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python"
CONFIG="${RUN_ROOT}/resolved_train_config.yaml"
MANIFEST="${RUN_ROOT}/metadata/metadata_corrected_direction_0719.csv"
SAMPLE_ID="mymotion__002"
TRAIN_GPUS="${TRAIN_GPUS:-1,2,4}"
ALLOWED_INFER_GPUS="${ALLOWED_INFER_GPUS:-1,2,4,5}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/reports"
printf '{"state":"training","gpu_ids":"%s"}\n' "${TRAIN_GPUS}" > "${RUN_ROOT}/reports/controller_state.json"

GPU_IDS="${TRAIN_GPUS}" "${RUN_ROOT}/scripts/launch_tiered_train.sh" \
  2>&1 | tee "${RUN_ROOT}/logs/train_corrected_tiers.log"

for tier in quick conservative recommended; do
  adapter="${RUN_ROOT}/${tier}/final_adapter/v2v_lora_and_source_patch.safetensors"
  test -s "${adapter}" || { echo "Missing ${tier} adapter: ${adapter}" >&2; exit 10; }
done

infer_gpu=""
IFS=',' read -r -a CANDIDATES <<< "${ALLOWED_INFER_GPUS}"
for gpu in "${CANDIDATES[@]}"; do
  [[ "${gpu}" == "0" ]] && continue
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( free_mib >= 70000 )); then
    infer_gpu="${gpu}"
    break
  fi
done
if [[ -z "${infer_gpu}" ]]; then
  echo "No nonzero GPU with at least 70 GiB free for inference" >&2
  exit 11
fi
printf '{"state":"inference","physical_gpu":"%s"}\n' "${infer_gpu}" > "${RUN_ROOT}/reports/controller_state.json"

export PYTHONPATH="${RUN_ROOT}/scripts:${FORMAL_ROOT}/environment/python_packages:${FORMAL_ROOT}/scripts:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
for tier in quick conservative recommended; do
  adapter="${RUN_ROOT}/${tier}/final_adapter/v2v_lora_and_source_patch.safetensors"
  output="${RUN_ROOT}/${tier}/validation"
  CUDA_VISIBLE_DEVICES="${infer_gpu}" "${PYTHON}" "${RUN_ROOT}/scripts/infer_v2v_comparison.py" \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --adapter "${adapter}" \
    --output-dir "${output}" \
    --sample-id "${SAMPLE_ID}" \
    --num-frames 33 --steps 30 --seed 42 \
    2>&1 | tee "${RUN_ROOT}/logs/infer_${tier}_${SAMPLE_ID}.log"
done

for tier in quick conservative recommended; do
  comparison="${RUN_ROOT}/${tier}/validation/videos/${SAMPLE_ID}_comparison.mp4"
  test -s "${comparison}" || { echo "Missing comparison: ${comparison}" >&2; exit 12; }
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,pix_fmt,codec_name \
    -of json "${comparison}" > "${RUN_ROOT}/${tier}/validation/reports/ffprobe_comparison.json"
done

printf '{"state":"complete","sample_id":"%s","quality_metrics_computed":false}\n' "${SAMPLE_ID}" > "${RUN_ROOT}/reports/controller_state.json"
echo "[complete] corrected tiered training and all three comparisons finished"
