#!/bin/bash
# ============================================================
# LiveAnimate 全参数微调 — FSDP 多机多卡
#
# 用法（每台机器分别执行，MASTER_IP 填 node0 的 IP）：
#   node0:  MASTER_IP=10.x.x.x MACHINE_RANK=0 bash train_fsdp_multinode.sh
#   node1:  MASTER_IP=10.x.x.x MACHINE_RANK=1 bash train_fsdp_multinode.sh
#
# 可选环境变量：
#   NUM_MACHINES      默认 2
#   NUM_GPU_PER_MACHINE  默认 8
#   MASTER_PORT       默认 29500
# ============================================================

set -e

MASTER_IP=${MASTER_IP:?"Please set MASTER_IP (node0's IP, same on ALL machines)"}
MASTER_PORT=${MASTER_PORT:-29500}
MACHINE_RANK=${MACHINE_RANK:?"Please set MACHINE_RANK (0 for master, 1+ for workers)"}
NUM_MACHINES=${NUM_MACHINES:-8}
NUM_GPU_PER_MACHINE=${NUM_GPU_PER_MACHINE:-8}
NUM_PROCESSES=$((NUM_MACHINES * NUM_GPU_PER_MACHINE))

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600

MODEL_DIR="/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B"
DATA_JSONL="/primus_chat_aigc_sh_oss/users/zhangyuxuan/T08temp/liveanimate_v4/merged.jsonl"
OUTPUT_DIR="./output_fsdp_lora_multinode_tsf"
TOKENIZER="${MODEL_DIR}/google/umt5-xxl"

MODEL_PATHS='[
  ["'"${MODEL_DIR}"'/diffusion_pytorch_model-00001-of-00004.safetensors",
   "'"${MODEL_DIR}"'/diffusion_pytorch_model-00002-of-00004.safetensors",
   "'"${MODEL_DIR}"'/diffusion_pytorch_model-00003-of-00004.safetensors",
   "'"${MODEL_DIR}"'/diffusion_pytorch_model-00004-of-00004.safetensors"],
  "'"${MODEL_DIR}"'/models_t5_umt5-xxl-enc-bf16.pth",
  "'"${MODEL_DIR}"'/Wan2.1_VAE.pth",
  "'"${MODEL_DIR}"'/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
]'

# Generate per-rank accelerate config
CONFIG_FILE="/tmp/accelerate_fsdp_rank${MACHINE_RANK}.yaml"
sed \
    "s/MASTER_IP_PLACEHOLDER/${MASTER_IP}/g; \
     s/machine_rank: 0/machine_rank: ${MACHINE_RANK}/g; \
     s/main_process_port: 29500/main_process_port: ${MASTER_PORT}/g; \
     s/num_machines: 8/num_machines: ${NUM_MACHINES}/g; \
     s/num_processes: 64/num_processes: ${NUM_PROCESSES}/g" \
    ./accelerate_config_fsdp_multinode.yaml > "${CONFIG_FILE}"

echo "=== Launching rank=${MACHINE_RANK}, master=${MASTER_IP}:${MASTER_PORT}, total_gpus=${NUM_PROCESSES} ==="

mkdir -p "${OUTPUT_DIR}"

accelerate launch \
    --config_file "${CONFIG_FILE}" \
    train_fsdp.py \
    --model_paths "${MODEL_PATHS}" \
    --tokenizer_path "${TOKENIZER}" \
    --dataset_path "${DATA_JSONL}" \
    --dataset_weights 1 \
    --max_num_frames 100 \
    --output_path "${OUTPUT_DIR}" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --learning_rate 1e-5 \
    --num_epochs 5 \
    --save_steps 500 \
    --vis_every 100 \
    --task "sft:causal_incontext_tsf" \
    --target_len 81 \
    --local_attn_size -1 \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 128 \
    2>&1 | tee "${OUTPUT_DIR}/train_rank${MACHINE_RANK}.log"
