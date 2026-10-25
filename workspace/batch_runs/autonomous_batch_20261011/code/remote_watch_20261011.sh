#!/usr/bin/env bash
set -u
B=/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011
M="$B/logs/remote_watch_20261011.jsonl"
mkdir -p "$B/logs"
while true; do
  a=$(grep -Eo '"done": [0-9]+' "$B/logs/semantic_full_a.log" 2>/dev/null | tail -n1 | awk '{print $2}' || true)
  b=$(grep -Eo '"done": [0-9]+' "$B/logs/semantic_full_b.log" 2>/dev/null | tail -n1 | awk '{print $2}' || true)
  pa=$(ps -p 884626 -o stat= 2>/dev/null | tr -d ' ' || true)
  pb=$(ps -p 885041 -o stat= 2>/dev/null | tr -d ' ' || true)
  mp4=$(find "$B/clips" -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  dirs=$(find "$B/clips" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
  disk=$(df -Pk /nvmedata 2>/dev/null | awk 'NR==2 {print $4}')
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)
  printf '{"ts":"%s","a_done":%s,"b_done":%s,"pid_a":"%s","pid_b":"%s","mp4":%s,"source_dirs":%s,"disk_kb":%s,"gpu":"%s"}\n' "$(date -Is)" "${a:-null}" "${b:-null}" "${pa:-stopped}" "${pb:-stopped}" "${mp4:-0}" "${dirs:-0}" "${disk:-null}" "$gpu" >> "$M"
  if [ "${ONESHOT:-0}" = 1 ]; then break; fi
  if [ -z "$pa" ] && [ -z "$pb" ]; then break; fi
  sleep 60
done