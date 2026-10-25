#!/usr/bin/env bash
set -u
ROOT=/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011
PY=/nvmedata/users/nina/conda_envs/sam2_mask_0808/bin/python
if [ ! -x "$PY" ]; then PY=/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python; fi
WORKER=$ROOT/code/qwen_semantic_worker.py
QUEUE=$ROOT/reports/semantic_second_pass_queue_20261011.jsonl
OUT=$ROOT/review_artifacts/qwen_semantic_full_overlap_20261011
LOGDIR=$ROOT/logs
SCHEDLOG=$LOGDIR/qwen_semantic_resume_scheduler.log
mkdir -p "$OUT" "$LOGDIR"
printf 'scheduler_started %s\n' "$(date --iso-8601=seconds)" >> "$SCHEDLOG"

progress_state() {
  local worker="$1"
  local p="$OUT/progress_worker_$(printf '%02d' "$worker").json"
  if [ ! -f "$p" ]; then echo pending; return; fi
  "$PY" - "$p" <<'PY'
import json,sys
try:
 d=json.load(open(sys.argv[1]))
 print('done' if int(d.get('completed',0)) >= int(d.get('assigned',1)) else 'pending')
except Exception:
 print('pending')
PY
}

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1; used=$2+0; util=$4+0; if (used < 4000 && util < 10) {gsub(/ /,"",g); print g; exit}}'
}

for worker in 1 0 2 3; do
  if [ "$(progress_state "$worker")" = done ]; then
    printf 'worker=%s already_complete %s\n' "$worker" "$(date --iso-8601=seconds)" >> "$SCHEDLOG"
    continue
  fi
  while true; do
    if [ "$(progress_state "$worker")" = done ]; then
      printf 'worker=%s completed %s\n' "$worker" "$(date --iso-8601=seconds)" >> "$SCHEDLOG"
      break
    fi
    existing=$(pgrep -f "$WORKER.*--worker $worker" || true)
    if [ -n "$existing" ]; then
      printf 'worker=%s already_running=%s %s\n' "$worker" "$existing" "$(date --iso-8601=seconds)" >> "$SCHEDLOG"
      sleep 60
      continue
    fi
    gpu=$(free_gpu || true)
    if [ -z "$gpu" ]; then
      printf 'worker=%s waiting_for_free_gpu %s\n' "$worker" "$(date --iso-8601=seconds)" >> "$SCHEDLOG"
      sleep 60
      continue
    fi
    log="$LOGDIR/qwen_semantic_resume_worker_${worker}.log"
    printf 'worker=%s launching_gpu=%s %s\n' "$worker" "$gpu" "$(date --iso-8601=seconds)" >> "$SCHEDLOG"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u "$WORKER" \
      --queue "$QUEUE" --out-dir "$OUT" --worker "$worker" --workers 4 --max-new-tokens 192 \
      > "$log" 2>&1 &
    pid=$!
    echo "$pid" > "$ROOT/locks/qwen_resume_worker_${worker}.pid"
    wait "$pid" || true
    printf 'worker=%s exited_pid=%s state=%s %s\n' "$worker" "$pid" "$(progress_state "$worker")" "$(date --iso-8601=seconds)" >> "$SCHEDLOG"
    sleep 10
  done
done
printf 'scheduler_finished %s\n' "$(date --iso-8601=seconds)" >> "$SCHEDLOG"
