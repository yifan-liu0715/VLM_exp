#!/usr/bin/env python3
"""Create per-clip and combined FFmpeg contact sheets for a smoke export."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.manifest)
    if len(rows) != 10:
        raise ValueError(f"expected 10 smoke rows, got {len(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strips: list[Path] = []
    for index, row in enumerate(rows, 1):
        video = Path(row["output_video"])
        duration = float(row["duration_sec"])
        strip = args.output_dir / f"{index:02d}_{row['clip_id']}_contact.jpg"
        label = f"{index:02d} {video.stem[-8:]}  {row.get('action_label', '')}"
        interval = max(duration / 5.0, 0.05)
        vf = (
            f"fps=1/{interval:.9f},scale=256:144:force_original_aspect_ratio=decrease,"
            "pad=256:144:(ow-iw)/2:(oh-ih)/2,"
            "drawbox=x=0:y=0:w=iw:h=24:color=black@0.65:t=fill,"
            f"drawtext=text='{label}':x=6:y=5:fontsize=12:fontcolor=white,"
            "tile=5x1"
        )
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", vf, "-frames:v", "1", str(strip)])
        strips.append(strip)

    combined = args.output_dir / "smoke_10_contact_sheet.jpg"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for strip in strips:
        command.extend(["-i", str(strip)])
    layout = "|".join(
        f"{(index % 2) * 1280}_{(index // 2) * 144}" for index in range(10)
    )
    command.extend(
        [
            "-filter_complex",
            f"xstack=inputs=10:layout={layout}:fill=black",
            "-frames:v",
            "1",
            str(combined),
        ]
    )
    run(command)
    print(json.dumps({"strip_count": len(strips), "combined": str(combined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
