#!/usr/bin/env bash
set -e
if test -f /nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/locks/semantic_pilot_10.lock; then echo LOCK_EXISTS; exit 2; fi
printf '%s\n' '{"run_id":"autonomous_batch_20261011","stage":"semantic_pilot_10","gpu":2,"started_at":"'"$(date --iso-8601=seconds)"'"}' > /nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/locks/semantic_pilot_10.lock
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/code:/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/deps/mediapipe /nvmedata/workspace2/conda_data/envs/LTX2_hxh/bin/python -u /nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/code/auto_semantic_batch.py --workers 1 --limit 10 --force > /nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/logs/semantic_pilot_10.log 2>&1 &
echo $! > /nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/locks/semantic_pilot_10.pid
cat /nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/locks/semantic_pilot_10.pid
