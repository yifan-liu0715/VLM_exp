#!/usr/bin/env bash
set -uo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803
TMP=/mnt/highspeed/cache/tmp/scail2_flashattn_0809

if [ "$(df --output=avail /nvmedata | tail -n 1)" -lt $((30 * 1024 * 1024)) ]; then
  echo "less than 30 GiB available on /nvmedata" >&2
  exit 90
fi

mkdir -p "$TMP"
cd "$ROOT" || exit 91
rm -f logs/install_flashattn.exit
"$ENV/bin/python" -m pip freeze > manifests/pip_freeze_before_flashattn.txt

export CUDA_VISIBLE_DEVICES=''
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0
export FLASH_ATTN_CUDA_ARCHS=80
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=8
export NVCC_THREADS=2
export TMPDIR="$TMP"

"$ENV/bin/python" -m pip install --no-build-isolation flash-attn==2.8.3.post1 \
  > logs/install_flashattn.log 2>&1
status=$?
if [ "$status" -eq 0 ]; then
  "$ENV/bin/python" -m pip show flash-attn >> logs/install_flashattn.log 2>&1
  "$ENV/bin/python" -m pip freeze > manifests/pip_freeze_after_flashattn.txt
fi
printf '%s\n' "$status" > logs/install_flashattn.exit
exit "$status"
