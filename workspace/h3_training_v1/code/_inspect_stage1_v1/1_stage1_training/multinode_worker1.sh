#!/bin/bash
# 用法：bash multinode_worker.sh <rank>
# 例如：bash multinode_worker.sh 1
# 在 worker 机器上后台启动 tmux session 并运行训练

MACHINE_RANK=${1:?"Usage: bash multinode_worker.sh <rank>  (e.g. 1, 2, ...)"}
MASTER_IP="10.11.221.50"   # 改成 master 实际 IP
SESSION="train_worker${MACHINE_RANK}"
WORK_DIR="/primus_xpfs_workspace_T08/zhangyuxuan/projects2/LiveAnimate_stage1_0430"
CONDA_SH="/primus_xpfs_workspace_T08/zhangyuxuan/tools/conda.sh"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" \
    "source ${CONDA_SH} on && \
     conda activate base && \
     cd ${WORK_DIR} && \
     MASTER_IP=${MASTER_IP} MACHINE_RANK=${MACHINE_RANK} bash train_fsdp_multinode_tf.sh; \
     echo '=== worker${MACHINE_RANK} done, press any key to exit ===' && read"

echo "Worker rank=${MACHINE_RANK} launched in tmux session '${SESSION}'"
echo "Attach with: tmux attach -t ${SESSION}"
