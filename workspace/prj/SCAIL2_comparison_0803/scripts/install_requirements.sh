#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803
REPO="$ROOT/code/SCAIL-2"
LOG="$ROOT/logs/install_requirements.log"
EXIT_FILE="$ROOT/logs/install_requirements.exit"

rm -f "$EXIT_FILE"
trap 'rc=$?; echo "$rc" > "$EXIT_FILE"' EXIT
{
  echo "START $(date -Is)"
  "$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$ENV/bin/python" -m pip install -r "$REPO/requirements.txt"
  "$ENV/bin/python" -m pip check
  conda list -p "$ENV" --explicit > "$ROOT/manifests/conda_explicit.txt"
  echo "COMPLETE $(date -Is)"
} >> "$LOG" 2>&1
