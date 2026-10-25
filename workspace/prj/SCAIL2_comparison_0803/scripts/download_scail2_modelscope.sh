#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803
DEST="$ROOT/weights/SCAIL-2"
LOG="$ROOT/logs/model_download.log"
EXIT_FILE="$ROOT/logs/model_download.exit"

rm -f "$EXIT_FILE"
trap 'rc=$?; echo "$rc" > "$EXIT_FILE"' EXIT

{
  echo "START $(date -Is)"
  echo "SOURCE modelscope:ZhipuAI/SCAIL-2"
  echo "REVISION master"
  df -h /nvmedata
  "$ENV/bin/modelscope" download ZhipuAI/SCAIL-2 \
    --revision master \
    --local-dir "$DEST" \
    --max-workers 4
  echo "DOWNLOADED $(date -Is)"
  find "$DEST" -type f -printf '%P\t%s\n' | sort
  echo "COMPLETE $(date -Is)"
} >> "$LOG" 2>&1
