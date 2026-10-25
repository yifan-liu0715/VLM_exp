#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
WEIGHTS="$PROJECT/pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B"
METADATA="$PROJECT/data_0713/v2v_quick_0713/train_targets_0713.csv"
DATASET="$PROJECT/data/dataset_video_train"
OUTPUT="$PROJECT/output_models/wan2.2_lora_quick_v2v_0713"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set explicitly." >&2
  exit 2
fi

for path in \
  "$METADATA" \
  "$WEIGHTS/models_t5_umt5-xxl-enc-bf16.pth" \
  "$WEIGHTS/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "$WEIGHTS/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "$WEIGHTS/diffusion_pytorch_model-00003-of-00003.safetensors" \
  "$WEIGHTS/Wan2.2_VAE.pth" \
  "$WEIGHTS/google/umt5-xxl"; do
  [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 2; }
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

echo "Experiment: native input_video V2V + target-domain LoRA (_0713)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Training targets: $METADATA"
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
  --num_epochs 4 \
  --save_steps 512 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "$OUTPUT" \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 16 \
  2>&1 | tee "$OUTPUT/training_0713.log"
