#!/usr/bin/env bash
set -uo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803
WAN=$ROOT/code/SCAIL-2-wan/SCAIL-2-5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5
INPUT=$ROOT/weights/SCAIL-2/model/1/fsdp2_rank_0000_checkpoint.pt
OUTPUT=$ROOT/weights/converted/SCAIL-2.safetensors
LOG=$ROOT/logs/convert_base.log
EXIT_FILE=$ROOT/logs/convert_base.exit

mkdir -p "$ROOT/weights/converted" "$ROOT/logs"
rm -f "$EXIT_FILE"
export CUDA_VISIBLE_DEVICES=""

"$ENV/bin/python" "$ROOT/scripts/convert_scail2_base_py29.py" \
  --official-converter "$WAN/convert.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --expected-sha256 d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f \
  >"$LOG" 2>&1
rc=$?
echo "$rc" >"$EXIT_FILE"
exit "$rc"
