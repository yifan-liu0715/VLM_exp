#!/usr/bin/env bash
set +e
printf "PILOT_REMOTE_DURATION\n"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004205501_0126_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004170726_0018_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004170621_0015_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004173539_0043_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004174250_0048_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004190023_0071_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004170124_0009_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004165648_0007_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004215044_0133_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
f='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004204650_0124_D.MP4'; dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1); printf '%s\t%s\n' "$f" "$dur"
