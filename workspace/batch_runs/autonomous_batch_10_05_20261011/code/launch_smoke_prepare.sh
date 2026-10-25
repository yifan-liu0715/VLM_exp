#!/usr/bin/env bash
set -euo pipefail
R=/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_10_05_20261011
if [ -f "$R/locks/smoke_prepare.pid" ]; then
  PREV=$(cat "$R/locks/smoke_prepare.pid")
  if ps -p "$PREV" -o args= 2>/dev/null | grep -q "$R/code/prepare_remote_pilot.py"; then
    echo "ALREADY_RUNNING $PREV"
    exit 0
  fi
fi
nohup env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$R/code:$R/deps/mediapipe" /nvmedata/workspace2/conda_data/envs/LTX2_hxh/bin/python -B -u "$R/code/prepare_remote_pilot.py" --root "$R" --inventory "$R/smoke_inventory.json" --stage smoke > "$R/logs/smoke_prepare.log" 2>&1 &
echo $! > "$R/locks/smoke_prepare.pid"
cat "$R/locks/smoke_prepare.pid"
