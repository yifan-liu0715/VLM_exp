#!/usr/bin/env python3
"""Independently validate all exported capture clips with ffprobe."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def probe_one(row: dict[str, Any]) -> dict[str, Any]:
    clip = Path(str(row["output_video"]))
    result: dict[str, Any] = {"clip_id": row.get("clip_id"), "path": str(clip)}
    if not clip.is_file() or clip.stat().st_size < 1000:
        result.update({"valid": False, "reason": "missing_or_empty"})
        return result
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames",
                "-of", "json", str(clip),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        video = [stream for stream in streams if stream.get("codec_type") == "video"]
        audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if len(video) != 1:
            result.update({"valid": False, "reason": "video_stream_count", "streams": streams})
            return result
        stream = video[0]
        duration = float(data.get("format", {}).get("duration") or 0.0)
        expected_width, expected_height = ((720, 1280) if row.get("height", 0) > row.get("width", 0) else (1280, 720))
        checks = {
            "duration": 3.92 <= duration <= 6.08,
            "codec": stream.get("codec_name") == "h264",
            "dimensions": (int(stream.get("width") or 0), int(stream.get("height") or 0)) == (expected_width, expected_height),
            "fps": stream.get("avg_frame_rate") == "24/1",
            "frames": int(stream.get("nb_frames") or 0) > 0,
            "no_audio": not audio,
        }
        result.update({
            "valid": all(checks.values()),
            "reason": None if all(checks.values()) else ",".join(key for key, value in checks.items() if not value),
            "duration_sec": duration,
            "codec": stream.get("codec_name"),
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "fps": stream.get("avg_frame_rate"),
            "nb_frames": int(stream.get("nb_frames") or 0),
            "audio_streams": len(audio),
        })
        return result
    except Exception as exc:
        result.update({"valid": False, "reason": repr(exc)})
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = load_rows(Path(args.manifest))
    workers = max(1, min(args.workers, len(rows) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(probe_one, rows))
    results.sort(key=lambda item: item["clip_id"])
    report = {
        "input_count": len(rows),
        "valid_count": sum(item.get("valid", False) for item in results),
        "invalid_count": sum(not item.get("valid", False) for item in results),
        "invalid": [item for item in results if not item.get("valid", False)],
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("input_count", "valid_count", "invalid_count")}))
    return 0 if report["invalid_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
