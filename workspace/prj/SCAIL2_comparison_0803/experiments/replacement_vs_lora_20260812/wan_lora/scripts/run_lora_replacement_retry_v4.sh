#!/usr/bin/env bash
set -euo pipefail

if [ "${SCAIL2_CLEAN_ENV:-0}" != "1" ]; then
  exec env -i \
    HOME=/home/student \
    USER=student \
    LOGNAME=student \
    SHELL=/bin/bash \
    LANG=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    SCAIL2_CLEAN_ENV=1 \
    /bin/bash "$0" "$@"
fi

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/experiments/replacement_vs_lora_20260812
PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
PYTHON=/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python
PYTHON_PACKAGES="$PROJECT/v2v_formal_0715/environment/python_packages"
ADAPTER="$PROJECT/v2v_reference_independent_posenc_scheme4_0729/recommended/final_adapter/v2v_lora_source_reference_posenc.safetensors"
EXPECTED_ADAPTER_SHA256=dece6fe3aa1de90166f861c4e10405ab32ceb4b1a78f61e9bac08af5f595908a
OUTPUT="$ROOT/wan_lora/outputs/lora_retry_v4.mp4"
MANIFEST="$ROOT/wan_lora/manifests/run_manifest_retry_v4.json"
LOG="$ROOT/wan_lora/logs/run_retry_v4.log"
GPU="${1:?pass one physical nonzero GPU id}"

if [ "$GPU" = "0" ]; then
  echo "physical GPU0 is forbidden" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "exactly one physical nonzero GPU is required" >&2
  exit 2
fi
if [ -e "$OUTPUT" ] || [ -e "$MANIFEST" ] || [ -e "$LOG" ]; then
  echo "refusing to overwrite existing LoRA retry v4 artifacts" >&2
  exit 3
fi

ACTIVE_PIDS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF' | paste -sd, -)
FREE_MIB=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if [ "$FREE_MIB" -lt 70000 ]; then
  echo "physical GPU $GPU has only ${FREE_MIB} MiB free" >&2
  exit 4
fi
FREE_BYTES=$(df -B1 --output=avail /nvmedata | tail -n 1 | tr -d ' ')
if [ "$FREE_BYTES" -lt 53687091200 ]; then
  echo "/nvmedata has less than 50 GiB free" >&2
  exit 5
fi

for path in \
  "$ROOT/wan_lora/scripts/infer_scheme4_lora_only.py" \
  "$ROOT/wan_lora/manifests/input_manifest.json" \
  "$ROOT/wan_lora/inputs/source.mp4" \
  "$ROOT/wan_lora/inputs/reference.png" \
  "$ADAPTER"; do
  if [ ! -f "$path" ]; then
    echo "required LoRA file is missing: $path" >&2
    exit 6
  fi
done

OBSERVED_ADAPTER_SHA256=$(sha256sum "$ADAPTER" | awk '{print $1}')
if [ "$OBSERVED_ADAPTER_SHA256" != "$EXPECTED_ADAPTER_SHA256" ]; then
  echo "LoRA adapter SHA256 mismatch" >&2
  exit 7
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/wan_lora/scripts:$PYTHON_PACKAGES:$PROJECT/v2v_reference_independent_posenc_scheme4_0729/scripts:$PROJECT/v2v_reference_independent_posenc_scheme4_0729/teacher_reference:$PROJECT"
exec > >(tee "$LOG") 2>&1
echo "physical_gpu=$GPU"
echo "variant=latest_lora"
echo "retry=v4"
echo "clean_environment=true"
echo "gpu_sharing_allowed=true"
echo "preexisting_compute_pids=${ACTIVE_PIDS:-none}"
echo "free_mib_at_launch=$FREE_MIB"
echo "adapter_sha256=$OBSERVED_ADAPTER_SHA256"
echo "python_faulthandler=true"
echo "cuda_launch_blocking=true"

"$PYTHON" -X faulthandler -u \
  "$ROOT/wan_lora/scripts/infer_scheme4_lora_only.py" \
  --input-manifest "$ROOT/wan_lora/manifests/input_manifest.json" \
  --adapter "$ADAPTER" \
  --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256" \
  --output-file "$OUTPUT" \
  --run-manifest "$MANIFEST"

echo "latest_lora_retry_v4_complete"
