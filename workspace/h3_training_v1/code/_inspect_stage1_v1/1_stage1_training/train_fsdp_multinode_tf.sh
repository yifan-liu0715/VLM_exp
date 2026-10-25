# #!/bin/bash
# # ============================================================
# # LiveAnimate Teacher Forcing — FSDP 多机多卡
# #
# # 用法（每台机器分别执行，MASTER_IP 填 node0 的 IP）：
# #   node0:  MASTER_IP=10.x.x.x MACHINE_RANK=0 bash train_fsdp_multinode_tf.sh
# #   node1:  MASTER_IP=10.x.x.x MACHINE_RANK=1 bash train_fsdp_multinode_tf.sh
# #
# # 可选环境变量：
# #   NUM_MACHINES      默认 2
# #   NUM_GPU_PER_MACHINE  默认 8
# #   MASTER_PORT       默认 29500
# # ============================================================

# set -e

# MASTER_IP=${MASTER_IP:?"Please set MASTER_IP (node0's IP, same on ALL machines)"}
# MASTER_PORT=${MASTER_PORT:-29500}
# MACHINE_RANK=${MACHINE_RANK:?"Please set MACHINE_RANK (0 for master, 1+ for workers)"}
# NUM_MACHINES=${NUM_MACHINES:-2}
# NUM_GPU_PER_MACHINE=${NUM_GPU_PER_MACHINE:-8}
# NUM_PROCESSES=$((NUM_MACHINES * NUM_GPU_PER_MACHINE))

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# # NCCL
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_TIMEOUT=1800

# MODEL_DIR="/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B"
# DATA_JSONL="/primus_xpfs_workspace_T08/zhangyuxuan/data/liveanimate_v4/merged_1_1.jsonl"
# OUTPUT_DIR="./output_fsdp_lora_multinode_tf_resume"
# TOKENIZER="${MODEL_DIR}/google/umt5-xxl"

# MODEL_PATHS='[
#   ["'"${MODEL_DIR}"'/diffusion_pytorch_model-00001-of-00004.safetensors",
#    "'"${MODEL_DIR}"'/diffusion_pytorch_model-00002-of-00004.safetensors",
#    "'"${MODEL_DIR}"'/diffusion_pytorch_model-00003-of-00004.safetensors",
#    "'"${MODEL_DIR}"'/diffusion_pytorch_model-00004-of-00004.safetensors"],
#   "'"${MODEL_DIR}"'/models_t5_umt5-xxl-enc-bf16.pth",
#   "'"${MODEL_DIR}"'/Wan2.1_VAE.pth",
#   "'"${MODEL_DIR}"'/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
# ]'

# # Generate per-rank accelerate config
# CONFIG_FILE="/tmp/accelerate_fsdp_rank${MACHINE_RANK}.yaml"
# sed \
#     "s/MASTER_IP_PLACEHOLDER/${MASTER_IP}/g; \
#      s/machine_rank: 0/machine_rank: ${MACHINE_RANK}/g; \
#      s/main_process_port: 29500/main_process_port: ${MASTER_PORT}/g; \
#      s/num_machines: 2/num_machines: ${NUM_MACHINES}/g; \
#      s/num_processes: 16/num_processes: ${NUM_PROCESSES}/g" \
#     ./accelerate_config_deepspeed_multinode.yaml > "${CONFIG_FILE}"

# echo "=== Launching rank=${MACHINE_RANK}, master=${MASTER_IP}:${MASTER_PORT}, total_gpus=${NUM_PROCESSES} ==="

# mkdir -p "${OUTPUT_DIR}"

# accelerate launch \
#     --config_file "${CONFIG_FILE}" \



#!/bin/bash
# ============================================================
# 性能诊断脚本 — DeepSpeed ZeRO-3 版本（torchrun 多节点启动）
#
# 用法：
#   机器 0: MASTER_IP=10.x.x.x MACHINE_RANK=0 bash train_deepspeed_profile.sh
#   机器 1: MASTER_IP=10.x.x.x MACHINE_RANK=1 bash train_deepspeed_profile.sh
# ============================================================

set -e

MASTER_IP=${MASTER_IP:?"Please set MASTER_IP"}
MASTER_PORT=${MASTER_PORT:-29500}
MACHINE_RANK=${MACHINE_RANK:-0}
NUM_MACHINES=${NUM_MACHINES:-2}
NUM_GPU_PER_MACHINE=${NUM_GPU_PER_MACHINE:-8}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600

MODEL_DIR="/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B"
DATA_JSONL="/primus_xpfs_workspace_T08/zhangyuxuan/data/liveanimate_v4/merged_1_1.jsonl"
OUTPUT_DIR="./output_fsdp_lora_multinode_tf_resume2"
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

echo "=== PROFILING MODE (DeepSpeed ZeRO-3): warmup=20, active=50 ==="
echo "=== node_rank=${MACHINE_RANK}, master=${MASTER_IP}:${MASTER_PORT}, nnodes=${NUM_MACHINES}, gpus_per_node=${NUM_GPU_PER_MACHINE} ==="

mkdir -p "${OUTPUT_DIR}"

# 使用 torchrun 启动，正确建立跨节点 torch.distributed 通信组
torchrun \
    --nnodes="${NUM_MACHINES}" \
    --nproc_per_node="${NUM_GPU_PER_MACHINE}" \
    --master_addr="${MASTER_IP}" \
    --master_port="${MASTER_PORT}" \
    --node_rank="${MACHINE_RANK}" \
    train_fsdp.py \
    --model_paths "${MODEL_PATHS}" \
    --tokenizer_path "${TOKENIZER}" \
    --dataset_path "${DATA_JSONL}" \
    --dataset_weights 1 \
    --max_num_frames 49 \
    --output_path "${OUTPUT_DIR}" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --save_steps 1000 \
    --vis_every 500 \
    --task "sft:causal_incontext_tf" \
    --initialize_model_on_cpu \
    --lora_checkpoint /primus_xpfs_workspace_T08/zhangyuxuan/projects2/LiveAnimate_stage1_0430/output_fsdp_lora_multinode_tf_resume/step-14000.safetensors \
    --target_len 49 \
    --local_attn_size -1 \
    --tf_clean_aug_max_timestep 0 \
    --tf_sync_noise_to_clean 0 \
    --vis_use_clean_x \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 128 \
    --use_deepspeed \
    2>&1 | tee "${OUTPUT_DIR}/train_rank${MACHINE_RANK}.log"
