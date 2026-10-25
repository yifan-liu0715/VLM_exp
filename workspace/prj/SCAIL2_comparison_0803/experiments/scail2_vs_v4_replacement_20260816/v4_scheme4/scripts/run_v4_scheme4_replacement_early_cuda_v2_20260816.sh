#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/experiments/scail2_vs_v4_replacement_20260816
INPUT_ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/experiments/replacement_vs_lora_20260812
PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
SCHEME4_ROOT="$PROJECT/v2v_reference_independent_posenc_scheme4_0729"
PYTHON=/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python
PYTHON_PACKAGES="$PROJECT/v2v_formal_0715/environment/python_packages"
ADAPTER="$SCHEME4_ROOT/recommended/final_adapter/v2v_lora_source_reference_posenc.safetensors"
EXPECTED_ADAPTER_SHA256=dece6fe3aa1de90166f861c4e10405ab32ceb4b1a78f61e9bac08af5f595908a
ENTRYPOINT="$ROOT/v4_scheme4/scripts/infer_scheme4_lora_only_early_cuda_v2.py"
INPUT_MANIFEST="$INPUT_ROOT/wan_lora/manifests/input_manifest.json"
OUTPUT="$ROOT/v4_scheme4/outputs/v4_scheme4_replacement.mp4"
MANIFEST="$ROOT/v4_scheme4/manifests/run_manifest.json"
GPU="${1:?pass one physical nonzero GPU id}"

if [ "$GPU" = "0" ]; then
  echo "physical GPU0 is forbidden" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "exactly one physical nonzero GPU is required" >&2
  exit 2
fi
if [ -e "$OUTPUT" ] || [ -e "$MANIFEST" ]; then
  echo "refusing to overwrite existing V4 output or manifest" >&2
  exit 3
fi

ACTIVE_PIDS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF')
FREE_MIB=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if [ -n "$ACTIVE_PIDS" ]; then
  echo "physical GPU $GPU has active compute processes: $ACTIVE_PIDS" >&2
  exit 4
fi
if [ "$FREE_MIB" -lt 70000 ]; then
  echo "physical GPU $GPU has only ${FREE_MIB} MiB free" >&2
  exit 4
fi
FREE_BYTES=$(df -B1 --output=avail /nvmedata | tail -n 1 | tr -d ' ')
if [ "$FREE_BYTES" -lt 53687091200 ]; then
  echo "/nvmedata has less than 50 GiB free" >&2
  exit 5
fi

for path in "$ENTRYPOINT" "$INPUT_MANIFEST" "$ADAPTER"
do
  if [ ! -f "$path" ]; then
    echo "required V4 file is missing: $path" >&2
    exit 6
  fi
done

OBSERVED_ADAPTER_SHA256=$(sha256sum "$ADAPTER" | awk '{print $1}')
if [ "$OBSERVED_ADAPTER_SHA256" != "$EXPECTED_ADAPTER_SHA256" ]; then
  echo "V4 adapter SHA256 mismatch" >&2
  exit 7
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/v4_scheme4/scripts:$PYTHON_PACKAGES:$SCHEME4_ROOT/scripts:$SCHEME4_ROOT/teacher_reference:$PROJECT:${PYTHONPATH:-}"
echo "physical_gpu=$GPU"
echo "variant=scheme4_v4"
echo "model_stack=Wan2.2-5B plus Scheme4 V4 adapter"
echo "launch_mode=direct_ssh_background"
echo "retry_reason=initialize_cuda_before_decord"
echo "free_mib_at_launch=$FREE_MIB"
echo "adapter_sha256=$OBSERVED_ADAPTER_SHA256"

"$PYTHON" -X faulthandler -u "$ENTRYPOINT" \
  --input-manifest "$INPUT_MANIFEST" \
  --adapter "$ADAPTER" \
  --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256" \
  --output-file "$OUTPUT" \
  --run-manifest "$MANIFEST"

echo "scheme4_v4_replacement_complete"
