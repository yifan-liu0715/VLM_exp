#!/usr/bin/env bash
set -euo pipefail

PROJECT=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
cd "$PROJECT"
out="$PROJECT/validation_results_0713/comparisons_0713"
mkdir -p "$out"

for id in mymotion__002 mymotion__012 mymotion__015; do
  src="data_0713/v2v_quick_0713/source_videos/${id}_source_degraded.mp4"
  tgt="data_0713/v2v_quick_0713/target_videos/${id}_target.mp4"
  base="validation_results_0713/${id}_base.mp4"
  lora="validation_results_0713/${id}_lora.mp4"
  ffmpeg -hide_banner -loglevel error -y \
    -i "$src" -i "$tgt" -i "$base" -i "$lora" \
    -filter_complex \
    "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[grid];[grid]drawtext=text='SOURCE-DEGRADED':x=12:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=5,drawtext=text='TARGET':x=588:y=12:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=5,drawtext=text='BASE-V2V':x=12:y=332:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=5,drawtext=text='LORA-V2V-STEP4096':x=588:y=332:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=5[v]" \
    -map "[v]" -an -c:v libx264 -crf 18 -preset medium \
    -pix_fmt yuv420p -movflags +faststart \
    "$out/${id}_v2v_comparison_0713.mp4"
done

for file in "$out"/*.mp4; do
  printf "%s " "$(basename "$file")"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,avg_frame_rate,nb_frames \
    -of csv=p=0 "$file"
done
