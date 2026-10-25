#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/dataset_h3_capture_v1
STAGE="$ROOT/incoming_20260918_increment_01"
ENV_NAME=v2v_train
MODEL="$ROOT/models/hand_landmarker.task"

test -s "$MODEL"
mkdir -p "$STAGE/hand_keypoints" "$STAGE/logs"

for shard in 00 01 02 03; do
    manifest="$STAGE/manifests/shards/shard_${shard}.jsonl"
    out="$STAGE/hand_keypoints/shard_${shard}"
    log="$STAGE/logs/hand_detection_${shard}.log"
    pid_file="$STAGE/logs/hand_detection_${shard}.pid"
    test -s "$manifest"
    mkdir -p "$out"
    env CUDA_VISIBLE_DEVICES= OPENCV_FFMPEG_READ_ATTEMPTS=20000 \
        OPENCV_FFMPEG_DECODE_ATTEMPTS=20000 \
        conda run --no-capture-output -n "$ENV_NAME" python \
        "$STAGE/run_capture_hand_detection.py" \
        --manifest "$manifest" \
        --clip-dir "$STAGE/clips" \
        --out "$out" \
        --model "$MODEL" \
        --sample-fps 24 \
        --opencv-threads 1 >"$log" 2>&1 &
    echo $! >"$pid_file"
done

echo "started_shards=4"
