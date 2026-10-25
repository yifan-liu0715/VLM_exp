#!/usr/bin/env bash

# 1. 强力清理旧进程（防止重启时端口被占用）
# 使用 pkill 更精准，防止 kill -9 报错
pkill -f train_distillation.py
pkill -f torchrun
sleep 2

# 2. 读取从命令行传进来的参数 (由 entrypoint 中的 $RANK $NNODES $MASTER_ADDR 传入)
# 语法：${1:-$RANK} 表示优先用第一个参数，没有就读环境变量 RANK
export NODE_RANK=${1:-$RANK}
export NNODES=${2:-$NNODES}
export MASTER_ADDR=${3:-$MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT:-29519}

# 3. 环境变量设置
export PYTHONPATH=$PYTHONPATH:. 
export CUDA_DEVICE_MAX_CONNECTIONS=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=25
export NCCL_DEBUG=INFO
export PRECOMPUTED_ONLY=1

# 自动检测 GPU 数
export NUM_ACCELERATORS=${NUM_ACCELERATORS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

echo "----------------------------------------------------------------"
echo "PHASE: Starting Training on Node ${NODE_RANK}"
echo "CONFIG: NNODES=${NNODES}, MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "----------------------------------------------------------------"

LOGDIR=logs_tf_3sm_3stsink_glr5e-6_clr5e-6_cn0_diffface_3steps
CONFIG=configs/v2v_dmd_training_sink_diffface.yaml
export VIS_SAVE_DIR="visualizations_${LOGDIR}"

mkdir -p $LOGDIR

# 5. 启动命令
PYTHON_EXE=python

# 重点：使用 --rdzv_endpoint 兼容性更好
${PYTHON_EXE} -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NUM_ACCELERATORS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_distillation_sink.py \
    --config_path $CONFIG \
    --logdir $LOGDIR \
    --wandb-save-dir wandb \
    --disable-wandb \
    2>&1 | tee -a ${LOGDIR}/output_node_${NODE_RANK}.log