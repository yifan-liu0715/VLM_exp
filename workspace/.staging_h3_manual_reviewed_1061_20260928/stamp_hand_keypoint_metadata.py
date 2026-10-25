#!/usr/bin/env python3
"""Bind hand-keypoint JSON files to the exact clip, model, and detector code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    from scripts.causal_clip_pipeline import (
        _hand_detection_signature as pipeline_hand_detection_signature,
    )
except ModuleNotFoundError:
    from causal_clip_pipeline import (  # type: ignore[no-redef]
        _hand_detection_signature as pipeline_hand_detection_signature,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def resolve_clip_path(clip_dir: Path, output_video: Any) -> Path:
    """Resolve both export-manifest absolute paths and legacy clip filenames."""
    candidate = Path(str(output_video))
    return candidate if candidate.is_absolute() else clip_dir / candidate


def legacy_detection_signature(
    clip_hash: str,
    model_hash: str,
    detector_hash: str,
    sample_fps: float,
    mediapipe_version: str,
) -> str:
    signature_payload = {
        "clip_sha256": clip_hash,
        "model_sha256": model_hash,
        "detector_sha256": detector_hash,
        "sample_fps": sample_fps,
        "mediapipe_version": mediapipe_version,
    }
    return hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def stamp(
    manifest: Path,
    clip_dir: Path,
    json_dir: Path,
    model: Path,
    detector: Path,
    mediapipe_version: str,
    output_manifest: Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(manifest)
    model_hash = sha256(model)
    detector_hash = sha256(detector)
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        clip_id = str(row["clip_id"])
        clip = resolve_clip_path(clip_dir, row["output_video"])
        json_path = json_dir / f"{clip_id}_hands.json"
        if not clip.is_file() or not json_path.is_file():
            errors.append(f"missing:{clip_id}")
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        summary = payload.get("summary")
        if not isinstance(frames, list) or not isinstance(summary, dict):
            errors.append(f"invalid_schema:{clip_id}")
            continue
        clip_hash = sha256(clip)
        declared_hash = row.get("output_sha256") or row.get("sha256")
        if clip_hash != str(declared_hash or ""):
            errors.append(f"clip_manifest_hash_mismatch:{clip_id}")
            continue
        frame_times: list[float] = []
        structure_ok = True
        for frame in frames:
            frame_times.append(float(frame.get("time_sec") or 0.0))
            hands = frame.get("hands")
            if not isinstance(hands, list) or int(frame.get("hand_count", -1)) != len(hands):
                structure_ok = False
                break
            for hand in hands:
                points = hand.get("points")
                if not isinstance(points, list) or len(points) != 21:
                    structure_ok = False
                    break
                if any(not isinstance(point, list) or len(point) != 3 for point in points):
                    structure_ok = False
                    break
        if not structure_ok or frame_times != sorted(frame_times):
            errors.append(f"invalid_frames:{clip_id}")
            continue
        sample_fps = float(
            summary.get("hand_sample_fps") or summary.get("sample_fps") or 0.0
        )
        row["output_sha256"] = clip_hash
        if row.get("boundary_signature") and row.get("phase_signature"):
            signature = pipeline_hand_detection_signature(
                row,
                model_hash,
                detector_hash,
                sample_fps,
                mediapipe_version,
            )
        else:
            signature = legacy_detection_signature(
                clip_hash,
                model_hash,
                detector_hash,
                sample_fps,
                mediapipe_version,
            )
        provenance = {
            "hand_detection_signature": signature,
            "hand_model_sha256": model_hash,
            "hand_detector_sha256": detector_hash,
            "hand_sample_fps": sample_fps,
            "mediapipe_version": mediapipe_version,
        }
        row.update(provenance)
        summary.update(row)
        summary.update(
            {
                "clip_path": str(clip.resolve()),
                "clip_sha256": clip_hash,
                "model_path": str(model.resolve()),
                "model_sha256": model_hash,
                "detector_path": str(detector.resolve()),
                "detector_sha256": detector_hash,
                "landmarker": f"MediaPipe Hand Landmarker {mediapipe_version}",
                "detection_signature": signature,
                "keypoint_schema": "mediapipe_hand_landmarker_21x3_v1",
                "keypoint_structure_valid": True,
                **provenance,
            }
        )
        payload["summary"] = summary
        atomic_write_text(
            json_path,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        summaries.append(summary)

    stamped_manifest = output_manifest or manifest
    write_jsonl(stamped_manifest, rows)
    report = {
        "manifest_count": len(rows),
        "completed_count": len(summaries),
        "error_count": len(errors),
        "errors": errors,
        "model_sha256": model_hash,
        "detector_sha256": detector_hash,
        "hand_model_sha256": model_hash,
        "hand_detector_sha256": detector_hash,
        "mediapipe_version": mediapipe_version,
        "output_manifest": str(stamped_manifest.resolve()),
        "samples": sorted(summaries, key=lambda row: str(row["clip_id"])),
    }
    atomic_write_text(
        json_dir / "summary.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--mediapipe-version", required=True)
    parser.add_argument(
        "--output-manifest",
        help="Write the stamped manifest here (defaults to updating --manifest in place)",
    )
    args = parser.parse_args()
    report = stamp(
        Path(args.manifest),
        Path(args.clip_dir),
        Path(args.json_dir),
        Path(args.model),
        Path(args.detector),
        args.mediapipe_version,
        output_manifest=Path(args.output_manifest) if args.output_manifest else None,
    )
    print(json.dumps({key: report[key] for key in ("manifest_count", "completed_count", "error_count")}))
    return 0 if report["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
