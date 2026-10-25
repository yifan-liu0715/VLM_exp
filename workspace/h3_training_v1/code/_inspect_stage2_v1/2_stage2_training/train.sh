# #!/bin/bash

# Project path and config
CONFIG=configs/v2v_dmd_training.yaml
LOGDIR=logs_tf_3sm_3stsink_glr5e-6_clr5e-6_cn0
WANDB_SAVE_DIR=wandb
export VIS_SAVE_DIR="visualizations_${LOGDIR}"

# Parse command line arguments
# NODE_RANK=${1}  
NODE_RANK=0

echo "CONFIG="$CONFIG
echo "NODE_RANK="$NODE_RANK

test -d "${LOGDIR}" || mkdir -p "${LOGDIR}"

# Suppress NCCL INFO logs
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 10.2.7.29
# 10.2.5.16
torchrun \
  --nnodes=1 \
  --node_rank=$NODE_RANK \
  --nproc_per_node=8 \
  --master_addr=10.5.189.61 \
  --master_port=29519 \
  train_distillation.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --disable-wandb \
  2>&1 | tee -a ${LOGDIR}/output_node_${NODE_RANK}.log