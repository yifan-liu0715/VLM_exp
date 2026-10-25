#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
DATASET=/nvmedata/workspace2/users/nina/dataset_video_train
METADATA="$PROJECT/data_splits/source_disjoint_v2/train.csv"
WEIGHTS="$PROJECT/pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B"
OUTPUT="$PROJECT/output_models/wan2.2_lora_formal_v2_r16_lr5e5"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set explicitly before launching training." >&2
  exit 2
fi

for required in \
  "$METADATA" \
  "$WEIGHTS/models_t5_umt5-xxl-enc-bf16.pth" \
  "$WEIGHTS/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "$WEIGHTS/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "$WEIGHTS/diffusion_pytorch_model-00003-of-00003.safetensors" \
  "$WEIGHTS/Wan2.2_VAE.pth" \
  "$WEIGHTS/google/umt5-xxl"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate v2v_train
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT"

MODEL_PATHS=$(printf '["%s",["%s","%s","%s"],"%s"]' \
  "$WEIGHTS/models_t5_umt5-xxl-enc-bf16.pth" \
  "$WEIGHTS/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "$WEIGHTS/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "$WEIGHTS/diffusion_pytorch_model-00003-of-00003.safetensors" \
  "$WEIGHTS/Wan2.2_VAE.pth")

echo "Starting formal Wan2.2 LoRA training on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Metadata: $METADATA"
echo "Output: $OUTPUT"

accelerate launch --num_processes 1 examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATASET" \
  --dataset_metadata_path "$METADATA" \
  --dataset_repeat 1 \
  --dataset_num_workers 2 \
  --data_file_keys video \
  --height 320 \
  --width 576 \
  --num_frames 33 \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "$WEIGHTS/google/umt5-xxl" \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --num_epochs 2 \
  --save_steps 750 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "$OUTPUT" \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 16 \
  --extra_inputs input_image \
  2>&1 | tee "$OUTPUT/training.log"
