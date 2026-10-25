#!/bin/bash
# ============================================================
# LiveAnimate 全参数微调 — FSDP 单机 8 卡
# 用法：bash train_fsdp.sh
# ============================================================

set -e

MODEL_DIR="/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B"
DATA_JSONL="/primus_xpfs_workspace_T08/zhangyuxuan/datasets/liveanimate_v3/merged.jsonl"
TOKENIZER="${MODEL_DIR}/google/umt5-xxl"

OUTPUT_DIR="./output_fsdplora_attn9_tsf_2500samples_lr5e-5"
test -d "${OUTPUT_DIR}" || mkdir -p "${OUTPUT_DIR}"

MODEL_PATHS='[
  ["'"${MODEL_DIR}"'/diffusion_pytorch_model-00001-of-00004.safetensors",
   "'"${MODEL_DIR}"'/diffusion_pytorch_model-00002-of-00004.safetensors",
   "'"${MODEL_DIR}"'/diffusion_pytorch_model-00003-of-00004.safetensors",
   "'"${MODEL_DIR}"'/diffusion_pytorch_model-00004-of-00004.safetensors"],
  "'"${MODEL_DIR}"'/models_t5_umt5-xxl-enc-bf16.pth",
  "'"${MODEL_DIR}"'/Wan2.1_VAE.pth",
  "'"${MODEL_DIR}"'/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
]'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
    --config_file ./accelerate_config_fsdp.yaml \
    train_fsdp.py \
    --model_paths "${MODEL_PATHS}" \
    --tokenizer_path "${TOKENIZER}" \
    --dataset_path "${DATA_JSONL}" \
    --dataset_weights 1 \
    --max_num_frames 81 \
    --output_path "${OUTPUT_DIR}" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --save_steps 500 \
    --vis_every 100 \
    --task "sft:causal_incontext_tsf" \
    --initialize_model_on_cpu \
    --target_len 76 \
    --local_attn_size -1 \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 128 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
