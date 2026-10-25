#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/experiments/replacement_vs_lora_20260812
SCAIL_SOURCE=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/code/SCAIL-2-wan/SCAIL-2-5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5
PYTHON=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python
WEIGHT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/weights/converted/SCAIL-2.safetensors
EXPECTED_WEIGHT_SHA256=8b844f026510f0b94d3209d66444b403a089ec179dc149d85513c13b2fa37118
OUTPUT="$ROOT/scail2/outputs/scail2_replacement.mp4"
LOG="$ROOT/scail2/logs/run.log"
GPU="${1:?pass physical GPU id 5}"

if [ "$GPU" != "5" ]; then
  echo "this shared runner is restricted to physical GPU5" >&2
  exit 2
fi
if [ -e "$OUTPUT" ] || [ -e "$LOG" ]; then
  echo "refusing to overwrite existing SCAIL output or log" >&2
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
  "$ROOT/scail2/inputs/reference.png" \
  "$ROOT/scail2/inputs/ref_mask.png" \
  "$ROOT/scail2/inputs/rendered_v2.mp4" \
  "$ROOT/scail2/inputs/replace_mask.mp4" \
  "$WEIGHT"; do
  if [ ! -f "$path" ]; then
    echo "required SCAIL file is missing: $path" >&2
    exit 6
  fi
done

OBSERVED_WEIGHT_SHA256=$(sha256sum "$WEIGHT" | awk '{print $1}')
if [ "$OBSERVED_WEIGHT_SHA256" != "$EXPECTED_WEIGHT_SHA256" ]; then
  echo "SCAIL weight SHA256 mismatch" >&2
  exit 7
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$SCAIL_SOURCE:${PYTHONPATH:-}"
exec > >(tee "$LOG") 2>&1
echo "physical_gpu=$GPU"
echo "mode=replacement"
echo "gpu_sharing_allowed=true"
echo "preexisting_compute_pids=${ACTIVE_PIDS:-none}"
echo "free_mib_at_launch=$FREE_MIB"
echo "weight_sha256=$OBSERVED_WEIGHT_SHA256"

cd "$SCAIL_SOURCE"
"$PYTHON" "$SCAIL_SOURCE/generate.py" \
  --model SCAIL-14B \
  --ckpt_dir /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/weights/SCAIL-2 \
  --scail_path "$WEIGHT" \
  --scail_config_path "$SCAIL_SOURCE/configs/config-14b.json" \
  --prompt "A slim young woman with long dark hair, wearing a black and white maid outfit and cat ears, performs a rhythmic dance. She starts with her back to the camera, turns fully around, and sways her hips while moving her arms. The setting is a minimalist room with white walls, a wooden chair, and a small framed picture." \
  --image "$ROOT/scail2/inputs/reference.png" \
  --mask_image "$ROOT/scail2/inputs/ref_mask.png" \
  --pose "$ROOT/scail2/inputs/rendered_v2.mp4" \
  --mask_video "$ROOT/scail2/inputs/replace_mask.mp4" \
  --replace_flag \
  --target_h 928 \
  --target_w 512 \
  --sample_solver unipc \
  --sample_steps 40 \
  --sample_shift 3.0 \
  --sample_guide_scale 5.0 \
  --segment_len 65 \
  --segment_overlap 5 \
  --base_seed 42 \
  --offload_model true \
  --save_file "$OUTPUT"

echo "scail2_replacement_complete"
