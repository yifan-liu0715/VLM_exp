#!/usr/bin/env python3
"""Manifest-driven causal event clipping and optional hand-keypoint extraction.

This pipeline is intentionally independent of the legacy fixed five-second
candidate selector. Every output lives below an explicit work root, and hand
detection runs only when requested by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


DEFAULT_PIPELINE_VERSION = "causal-boundary-v1"
MIN_DURATION = Decimal("2")
MAX_DURATION = Decimal("20")
MAX_SMOKE_EXPORTS = 10
CLIP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PHASE_FIELDS = (
    "pre_start_sec_rel",
    "pre_end_sec_rel",
    "action_start_sec_rel",
    "action_end_sec_rel",
    "response_start_sec_rel",
    "response_end_sec_rel",
    "post_start_sec_rel",
    "post_end_sec_rel",
)


class ProposalError(ValueError):
    """Raised when an event proposal cannot form a valid causal clip."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            raise ProposalError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ProposalError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ProposalError(f"{field} must be a finite number")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ProposalError(f"{field} must be a finite number") from exc
    if not number.is_finite():
        raise ProposalError(f"{field} must be a finite number")
    return number


def canonical_decimal(value: Any) -> str:
    number = _decimal(value, "boundary")
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_sha256(cache: dict[Path, str], path: Path) -> str:
    if path not in cache:
        cache[path] = sha256_file(path)
    return cache[path]


def boundary_signature(
    source_sha256: str,
    source_start_sec: Any,
    source_end_sec: Any,
    pipeline_version: str,
) -> str:
    payload = "\n".join(
        (
            str(source_sha256),
            canonical_decimal(source_start_sec),
            canonical_decimal(source_end_sec),
            str(pipeline_version),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def phase_signature(row: dict[str, Any]) -> str:
    payload = [str(row.get("boundary_signature", ""))]
    payload.extend(canonical_decimal(row.get(name)) for name in PHASE_FIELDS)
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def _portable_source_rel(value: Any) -> str:
    text = str(value or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or (relative.parts and ":" in relative.parts[0])
    ):
        raise ProposalError(f"source_rel must be a safe relative path: {value!r}")
    return relative.as_posix()


def resolve_source_video(
    row: dict[str, Any], source_root: Path | None = None
) -> tuple[Path, str | None]:
    if source_root is not None:
        root = Path(source_root).expanduser().resolve()
        if row.get("source_rel"):
            source_rel = _portable_source_rel(row["source_rel"])
            source = root.joinpath(*PurePosixPath(source_rel).parts).resolve()
        else:
            raw_source = row.get("source_video")
            if not raw_source:
                raise ProposalError("source_video or source_rel is required")
            source = Path(str(raw_source)).expanduser().resolve()
            try:
                source_rel = source.relative_to(root).as_posix()
            except ValueError as exc:
                raise ProposalError(
                    f"source_video is outside source_root and source_rel is missing: {source}"
                ) from exc
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ProposalError(f"source_rel escapes source_root: {source}") from exc
        return source, source_rel

    raw_source = row.get("source_video")
    if not raw_source:
        raise ProposalError("source_video is required without source_root")
    source = Path(str(raw_source)).expanduser().resolve()
    source_rel = (
        _portable_source_rel(row["source_rel"])
        if row.get("source_rel")
        else None
    )
    return source, source_rel


def _phase_value(row: dict[str, Any], relative_name: str, source_start: Decimal) -> Decimal:
    candidates = [relative_name]
    if relative_name == "action_start_sec_rel":
        candidates.append("hand_onset_sec_rel")
    for name in candidates:
        if name in row:
            return _decimal(row[name], name)

    absolute_candidates = [relative_name.replace("_rel", "_abs")]
    if relative_name == "action_start_sec_rel":
        absolute_candidates.append("hand_onset_sec_abs")
    for name in absolute_candidates:
        if name in row:
            return _decimal(row[name], name) - source_start
    raise ProposalError(f"missing phase boundary: {relative_name}")


def _validate_phase_values(values: dict[str, Decimal], duration: Decimal) -> None:
    for name, value in values.items():
        if value < 0 or value > duration:
            raise ProposalError(f"{name} must be within clip duration")
    for prefix in ("pre", "action", "response", "post"):
        start = values[f"{prefix}_start_sec_rel"]
        end = values[f"{prefix}_end_sec_rel"]
        if start > end:
            raise ProposalError(f"{prefix} start must not exceed {prefix} end")
    if values["response_start_sec_rel"] < values["action_start_sec_rel"]:
        raise ProposalError("response must not start before action")


def normalize_proposals(
    rows: Iterable[dict[str, Any]],
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    *,
    source_root: Path | None = None,
    max_duration: Any = MAX_DURATION,
) -> list[dict[str, Any]]:
    if not pipeline_version.strip():
        raise ProposalError("pipeline_version must not be empty")
    maximum_duration = _decimal(max_duration, "max_duration")
    if maximum_duration < MIN_DURATION:
        raise ProposalError(f"max_duration must be at least {MIN_DURATION} seconds")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_hashes: dict[Path, str] = {}
    source_probes: dict[Path, dict[str, Any]] = {}
    for index, original in enumerate(rows, 1):
        row = dict(original)
        clip_id = str(row.get("clip_id", ""))
        if not CLIP_ID_RE.fullmatch(clip_id):
            raise ProposalError(f"row {index}: invalid clip_id: {clip_id!r}")
        if clip_id in seen_ids:
            raise ProposalError(f"duplicate clip_id: {clip_id}")
        seen_ids.add(clip_id)

        source, source_rel = resolve_source_video(row, source_root)
        if not source.is_file():
            raise ProposalError(f"{clip_id}: missing source_video: {source}")
        start = _decimal(row.get("source_start_sec"), "source_start_sec")
        end = _decimal(row.get("source_end_sec"), "source_end_sec")
        duration = end - start
        if start < 0 or duration < MIN_DURATION or duration > maximum_duration:
            raise ProposalError(
                f"{clip_id}: duration must be between {MIN_DURATION} and {maximum_duration} seconds"
            )
        if source not in source_probes:
            try:
                source_probes[source] = probe_video(source)
            except Exception as exc:
                raise ProposalError(f"{clip_id}: source video is not decodable: {exc}") from exc
        source_info = source_probes[source]
        source_duration = _decimal(
            source_info.get("duration_sec"), "source_duration_sec"
        )
        if end > source_duration + Decimal("0.001"):
            raise ProposalError(
                f"{clip_id}: source_end_sec exceeds source duration {source_duration}"
            )
        camera_motion = str(row.get("camera_motion", "")).strip()
        review_status = str(row.get("review_status", "")).strip()
        if not camera_motion:
            raise ProposalError(f"{clip_id}: camera_motion is required")
        if not review_status:
            raise ProposalError(f"{clip_id}: review_status is required")

        phases = {
            name: _phase_value(row, name, start)
            for name in PHASE_FIELDS
        }
        _validate_phase_values(phases, duration)
        source_sha = _cached_sha256(source_hashes, source)
        supplied_sha = str(row.get("source_sha256", source_sha))
        if supplied_sha != source_sha:
            raise ProposalError(f"{clip_id}: source_sha256 does not match source file")
        start_text = canonical_decimal(start)
        end_text = canonical_decimal(end)
        signature = boundary_signature(source_sha, start_text, end_text, pipeline_version)

        result = _jsonable(row)
        result.update(
            {
                "clip_id": clip_id,
                "source_video": str(source),
                "source_rel": source_rel,
                "source_name": str(row.get("source_name") or source.name),
                "source_duration_sec": float(source_duration),
                "source_start_sec": float(start),
                "source_end_sec": float(end),
                "duration_sec": float(duration),
                "max_duration_sec": float(maximum_duration),
                "camera_motion": camera_motion,
                "review_status": review_status,
                "proposal_origin": str(row.get("proposal_origin") or "unknown"),
                "source_sha256": source_sha,
                "pipeline_version": pipeline_version,
                "boundary_start_sec_text": start_text,
                "boundary_end_sec_text": end_text,
                "boundary_signature": signature,
                "boundary_signature_algorithm": "sha256(source_sha256,start,end,pipeline_version)",
            }
        )
        result.update({name: float(value) for name, value in phases.items()})
        result["phase_signature"] = phase_signature(result)
        normalized.append(result)
    return normalized


def probe_video(path: Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_name,codec_type,width,height,avg_frame_rate,nb_frames:stream_disposition=attached_pic",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    videos = [
        stream
        for stream in streams
        if stream.get("codec_type") == "video"
        and not (stream.get("disposition") or {}).get("attached_pic")
    ]
    if not videos:
        raise RuntimeError(f"no primary video stream: {path}")
    stream = videos[0]
    return {
        "duration_sec": float(data.get("format", {}).get("duration") or 0.0),
        "size_bytes": int(data.get("format", {}).get("size") or path.stat().st_size),
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": stream.get("avg_frame_rate"),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "audio_streams": sum(stream.get("codec_type") == "audio" for stream in streams),
    }


def verify_full_decode(path: Path, ffmpeg_bin: str = "ffmpeg") -> None:
    subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=900,
    )


def _target_dimensions(row: dict[str, Any], source_info: dict[str, Any]) -> tuple[int, int]:
    width = int(row.get("width") or source_info.get("width") or 0)
    height = int(row.get("height") or source_info.get("height") or 0)
    return (720, 1280) if height > width else (1280, 720)


def _export_checks(
    info: dict[str, Any], duration: float, target_width: int, target_height: int
) -> dict[str, bool]:
    return {
        "duration": abs(info["duration_sec"] - duration) <= 0.08,
        "codec": info["codec"] == "h264",
        "dimensions": (info["width"], info["height"])
        == (target_width, target_height),
        "fps": info["fps"] == "24/1",
        "frames": info["nb_frames"] > 0,
        "no_audio": info["audio_streams"] == 0,
    }


def _export_signature(
    row: dict[str, Any], crf: int, target_width: int, target_height: int
) -> str:
    profile = {
        "boundary_signature": row["boundary_signature"],
        "phase_signature": row["phase_signature"],
        "boundary_policy_sha256": row.get("boundary_policy_sha256"),
        "codec": "libx264",
        "preset": "veryfast",
        "crf": int(crf),
        "pix_fmt": "yuv420p",
        "fps": "24/1",
        "width": target_width,
        "height": target_height,
        "audio": False,
    }
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def export_rows(
    rows: Iterable[dict[str, Any]],
    work_root: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    crf: int = 20,
    source_root: Path | None = None,
    write_manifest: bool = True,
) -> list[dict[str, Any]]:
    work_root = Path(work_root)
    clip_dir = work_root / "clips"
    state_dir = work_root / "state" / "exports"
    clip_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    source_hashes: dict[Path, str] = {}
    source_probes: dict[Path, dict[str, Any]] = {}
    for row in rows:
        clip_id = str(row["clip_id"])
        output_path = clip_dir / f"{clip_id}.mp4"
        sidecar_path = state_dir / f"{clip_id}.json"
        try:
            manifest_errors = _manifest_row_errors(row)
            if manifest_errors:
                raise ProposalError(
                    "invalid normalized manifest row: " + ",".join(manifest_errors)
                )
            source, source_rel = resolve_source_video(row, source_root)
            if not source.is_file():
                raise ProposalError(f"missing source_video: {source}")
            actual_source_sha = _cached_sha256(source_hashes, source)
            if actual_source_sha != row.get("source_sha256"):
                raise ProposalError("source_sha256 does not match source file")
            start_decimal = _decimal(row["source_start_sec"], "source_start_sec")
            end_decimal = _decimal(row["source_end_sec"], "source_end_sec")
            duration_decimal = end_decimal - start_decimal
            duration = float(duration_decimal)
            if source not in source_probes:
                source_probes[source] = probe_video(source, ffprobe_bin)
            source_info = source_probes[source]
            if end_decimal > _decimal(
                source_info["duration_sec"], "source_duration_sec"
            ) + Decimal("0.001"):
                raise ProposalError("source_end_sec exceeds actual source duration")
            target_width, target_height = _target_dimensions(row, source_info)
            export_signature = _export_signature(
                row, crf, target_width, target_height
            )
            sidecar = _load_json(sidecar_path)
            if (
                sidecar
                and sidecar.get("boundary_signature") == row["boundary_signature"]
                and sidecar.get("export_signature") == export_signature
                and output_path.is_file()
                and sidecar.get("output_sha256") == sha256_file(output_path)
            ):
                output_info = probe_video(output_path, ffprobe_bin)
                verify_full_decode(output_path, ffmpeg_bin)
                output_info["full_decode_verified"] = True
                checks = _export_checks(
                    output_info, duration, target_width, target_height
                )
                if all(checks.values()):
                    results.append(
                        {
                            **row,
                            "resolved_source_video": str(source),
                            "source_rel": source_rel,
                            "output_video": str(output_path.resolve()),
                            "export_status": "ok",
                            "cache_status": "reused",
                            "export_signature": export_signature,
                            "output_sha256": sidecar["output_sha256"],
                            "output": output_info,
                        }
                    )
                    continue

            temporary = clip_dir / f".{clip_id}.{os.getpid()}.tmp.mp4"
            vf = (
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,fps=24"
            )
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    row["boundary_start_sec_text"],
                    "-i",
                    str(source),
                    "-t",
                    canonical_decimal(duration_decimal),
                    "-an",
                    "-vf",
                    vf,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                check=True,
                timeout=900,
            )
            output_info = probe_video(temporary, ffprobe_bin)
            verify_full_decode(temporary, ffmpeg_bin)
            output_info["full_decode_verified"] = True
            checks = _export_checks(output_info, duration, target_width, target_height)
            if not all(checks.values()):
                temporary.unlink(missing_ok=True)
                failed = ",".join(name for name, value in checks.items() if not value)
                raise RuntimeError(f"export validation failed: {failed}")
            os.replace(temporary, output_path)
            output_sha = sha256_file(output_path)
            sidecar_payload = {
                "clip_id": clip_id,
                "boundary_signature": row["boundary_signature"],
                "export_signature": export_signature,
                "output_sha256": output_sha,
                "output": output_info,
            }
            _atomic_write_text(
                sidecar_path,
                json.dumps(sidecar_payload, ensure_ascii=False, indent=2),
            )
            results.append(
                {
                    **row,
                    "resolved_source_video": str(source),
                    "source_rel": source_rel,
                    "output_video": str(output_path.resolve()),
                    "export_status": "ok",
                    "cache_status": "generated",
                    "export_signature": export_signature,
                    "output_sha256": output_sha,
                    "output": output_info,
                }
            )
        except Exception as exc:
            results.append(
                {
                    **row,
                    "output_video": str(output_path.resolve()),
                    "export_status": "error",
                    "cache_status": "failed",
                    "export_error": repr(exc),
                }
            )
    if write_manifest:
        write_jsonl(work_root / "manifests" / "export_manifest.jsonl", results)
    return results


def validate_keypoint_payload(
    payload: dict[str, Any],
    row: dict[str, Any],
    *,
    require_provenance: bool = False,
) -> list[str]:
    reasons: list[str] = []
    summary = payload.get("summary")
    summary_dict = summary if isinstance(summary, dict) else {}
    if not isinstance(summary, dict):
        reasons.append("keypoint_missing_summary")
    else:
        if summary.get("clip_id") != row.get("clip_id"):
            reasons.append("keypoint_clip_id_mismatch")
        if summary.get("boundary_signature") != row.get("boundary_signature"):
            reasons.append("keypoint_boundary_signature_mismatch")
        if row.get("hand_detection_signature") and summary.get(
            "hand_detection_signature"
        ) != row.get("hand_detection_signature"):
            reasons.append("keypoint_detection_signature_mismatch")
        if row.get("hand_model_sha256") and summary.get(
            "hand_model_sha256"
        ) != row.get("hand_model_sha256"):
            reasons.append("keypoint_model_sha256_mismatch")
        if row.get("hand_detector_sha256") and summary.get(
            "hand_detector_sha256"
        ) != row.get("hand_detector_sha256"):
            reasons.append("keypoint_detector_sha256_mismatch")
        if row.get("mediapipe_version") and summary.get(
            "mediapipe_version"
        ) != row.get("mediapipe_version"):
            reasons.append("keypoint_mediapipe_version_mismatch")
    provenance_fields = (
        "hand_detection_signature",
        "hand_model_sha256",
        "hand_detector_sha256",
        "mediapipe_version",
        "hand_sample_fps",
    )
    if require_provenance and any(
        not row.get(name) or not summary_dict.get(name)
        for name in provenance_fields
    ):
        reasons.append("keypoint_missing_provenance")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        reasons.append("keypoint_empty_frames")
        return reasons
    previous_time = -math.inf
    valid_times: list[float] = []
    # Keypoint timestamps belong to the encoded CFR clip.  The signed source
    # boundary can differ by up to one frame after ffmpeg quantizes duration.
    output_info = row.get("output") if isinstance(row.get("output"), dict) else {}
    duration = float(output_info.get("duration_sec") or row.get("duration_sec") or 0.0)
    point_error = False
    for frame in frames:
        if not isinstance(frame, dict):
            reasons.append("keypoint_bad_frame")
            continue
        time_sec = frame.get("time_sec")
        if (
            not isinstance(time_sec, (int, float))
            or isinstance(time_sec, bool)
            or not math.isfinite(float(time_sec))
        ):
            reasons.append("keypoint_bad_timestamp")
        else:
            numeric_time = float(time_sec)
            if numeric_time < -1e-6 or numeric_time > duration + 0.08:
                reasons.append("keypoint_timestamp_out_of_range")
            if numeric_time < previous_time:
                reasons.append("keypoint_timestamps_not_monotonic")
            previous_time = numeric_time
            valid_times.append(numeric_time)
        hands = frame.get("hands")
        if not isinstance(hands, list) or frame.get("hand_count") != len(hands):
            reasons.append("keypoint_hand_count_mismatch")
            continue
        for hand in hands:
            points = hand.get("points") if isinstance(hand, dict) else None
            if not isinstance(points, list) or len(points) != 21:
                point_error = True
                continue
            for point in points:
                if (
                    not isinstance(point, list)
                    or len(point) != 3
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        for value in point
                    )
                ):
                    reasons.append("keypoint_bad_xyz")
                    break
    if point_error:
        reasons.append("keypoint_not_21_points")
    if valid_times and duration > 0:
        raw_summary_fps = summary_dict.get("hand_sample_fps") or summary_dict.get(
            "sample_fps"
        )
        raw_row_fps = row.get("hand_sample_fps") or raw_summary_fps or 24.0
        try:
            sample_fps = float(raw_summary_fps or raw_row_fps)
            row_sample_fps = float(raw_row_fps)
            if (
                not math.isfinite(sample_fps)
                or sample_fps <= 0
                or not math.isfinite(row_sample_fps)
                or row_sample_fps <= 0
            ):
                raise ValueError("non-positive or non-finite sample rate")
            if abs(sample_fps - row_sample_fps) > 1e-6:
                reasons.append("keypoint_sample_fps_mismatch")
        except (TypeError, ValueError, OverflowError):
            reasons.append("keypoint_sample_fps_invalid")
            sample_fps = 24.0
        end_tolerance = max(0.08, 2.0 / max(sample_fps, 1e-6))
        if valid_times[0] > 0.08 or valid_times[-1] < duration - end_tolerance:
            reasons.append("keypoint_timeline_incomplete")
        expected_frames = int(round(duration * sample_fps))
        if abs(len(frames) - expected_frames) > 1:
            reasons.append("keypoint_frame_count_mismatch")
        if summary_dict.get("frames_sampled") != len(frames):
            reasons.append("keypoint_summary_frame_count_mismatch")
        max_gap = 1.5 / max(sample_fps, 1e-6)
        if any(
            current - previous > max_gap
            for previous, current in zip(valid_times, valid_times[1:])
        ):
            reasons.append("keypoint_sampling_gap")
    return list(dict.fromkeys(reasons))


def _valid_matching_keypoints(path: Path, row: dict[str, Any]) -> bool:
    payload = _load_json(path)
    return payload is not None and not validate_keypoint_payload(
        payload, row, require_provenance=True
    )


def _runtime_mediapipe_version(python_executable: str) -> str:
    proc = subprocess.run(
        [
            python_executable,
            "-c",
            "import mediapipe as mp; print(mp.__version__)",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    version = proc.stdout.strip()
    if not version:
        raise RuntimeError("MediaPipe runtime did not report __version__")
    return version


def _hand_detection_signature(
    row: dict[str, Any],
    model_sha256: str,
    detector_sha256: str,
    sample_fps: float,
    mediapipe_version: str,
) -> str:
    profile = {
        "boundary_signature": row["boundary_signature"],
        "phase_signature": row["phase_signature"],
        "output_sha256": row["output_sha256"],
        "model_sha256": model_sha256,
        "detector_sha256": detector_sha256,
        "sample_fps": canonical_decimal(sample_fps),
        "mediapipe_version": mediapipe_version,
        "review_status": row.get("review_status"),
        "boundary_policy_sha256": row.get("boundary_policy_sha256"),
        "approved_trial_ids": row.get("approved_trial_ids", []),
    }
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def run_hand_detection_rows(
    rows: Iterable[dict[str, Any]],
    work_root: Path,
    detector_script: Path,
    model_path: Path,
    *,
    python_executable: str = sys.executable,
    sample_fps: float = 24.0,
    mediapipe_version: str | None = None,
) -> dict[str, Any]:
    work_root = Path(work_root)
    detector_script = Path(detector_script)
    model_path = Path(model_path)
    if not detector_script.is_file():
        raise FileNotFoundError(detector_script)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    runtime_version = mediapipe_version or _runtime_mediapipe_version(
        python_executable
    )
    model_sha = sha256_file(model_path)
    detector_sha = sha256_file(detector_script)
    hand_dir = work_root / "hand_keypoints"
    hand_dir.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    usable_rows = [
        row for row in materialized if row.get("export_status") == "ok"
    ]
    pending: list[dict[str, Any]] = []
    reused = 0
    for row in usable_rows:
        row.update(
            {
                "hand_model_sha256": model_sha,
                "hand_detector_sha256": detector_sha,
                "hand_sample_fps": sample_fps,
                "mediapipe_version": runtime_version,
                "hand_detection_signature": _hand_detection_signature(
                    row,
                    model_sha,
                    detector_sha,
                    sample_fps,
                    runtime_version,
                ),
            }
        )
        output = hand_dir / f"{row['clip_id']}_hands.json"
        if _valid_matching_keypoints(output, row):
            reused += 1
        else:
            pending.append(row)
    pending_manifest = work_root / "manifests" / "hand_detection_pending.jsonl"
    write_jsonl(pending_manifest, pending)
    write_jsonl(work_root / "manifests" / "export_manifest.jsonl", materialized)
    if not pending:
        report = {
            "input_count": len(usable_rows),
            "generated_count": 0,
            "reused_count": reused,
            "error_count": 0,
            "mediapipe_version": runtime_version,
            "hand_model_sha256": model_sha,
            "hand_detector_sha256": detector_sha,
        }
        _atomic_write_text(
            work_root / "reports" / "hand_detection.json",
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return report

    temporary_root = work_root / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="hand_detection_", dir=temporary_root))
    staging_out = staging / "out"
    staging_out.mkdir()
    command = [
        python_executable,
        str(detector_script),
        "--manifest",
        str(pending_manifest),
        "--clip-dir",
        str((work_root / "clips").resolve()),
        "--out",
        str(staging_out),
        "--model",
        str(model_path),
        "--sample-fps",
        str(sample_fps),
        "--opencv-threads",
        "1",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"hand detector failed ({proc.returncode}); staging retained at {staging}: "
            f"{proc.stderr[-1000:]}"
        )
    generated: list[tuple[Path, Path]] = []
    errors: list[dict[str, str]] = []
    for row in pending:
        staged = staging_out / f"{row['clip_id']}_hands.json"
        payload = _load_json(staged)
        if payload is not None and isinstance(payload.get("summary"), dict):
            payload["summary"].update(
                {
                    "landmarker": f"MediaPipe Hand Landmarker {runtime_version}",
                    "mediapipe_version": runtime_version,
                    "hand_model_sha256": model_sha,
                    "hand_detector_sha256": detector_sha,
                    "hand_sample_fps": sample_fps,
                    "hand_detection_signature": row[
                        "hand_detection_signature"
                    ],
                }
            )
            _atomic_write_text(
                staged,
                json.dumps(payload, ensure_ascii=False),
            )
        reasons = (
            ["keypoint_missing_output"]
            if payload is None
            else validate_keypoint_payload(
                payload, row, require_provenance=True
            )
        )
        if reasons:
            errors.extend(
                {"clip_id": str(row["clip_id"]), "reason": reason}
                for reason in reasons
            )
        else:
            generated.append((staged, hand_dir / staged.name))
    if errors:
        raise RuntimeError(
            f"hand detector produced invalid output; staging retained at {staging}: {errors}"
        )
    for source, destination in generated:
        os.replace(source, destination)
    shutil.rmtree(staging)
    report = {
        "input_count": len(usable_rows),
        "generated_count": len(generated),
        "reused_count": reused,
        "error_count": 0,
        "mediapipe_version": runtime_version,
        "hand_model_sha256": model_sha,
        "hand_detector_sha256": detector_sha,
    }
    _atomic_write_text(
        work_root / "reports" / "hand_detection.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    return report


def _manifest_row_errors(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    clip_id = str(row.get("clip_id", ""))
    if not CLIP_ID_RE.fullmatch(clip_id):
        reasons.append("manifest_invalid_clip_id")
    if not str(row.get("camera_motion", "")).strip():
        reasons.append("manifest_missing_camera_motion")
    if not str(row.get("review_status", "")).strip():
        reasons.append("manifest_missing_review_status")
    try:
        start = _decimal(row.get("source_start_sec"), "source_start_sec")
        end = _decimal(row.get("source_end_sec"), "source_end_sec")
        duration = end - start
        maximum_duration = _decimal(
            row.get("max_duration_sec", MAX_DURATION),
            "max_duration_sec",
        )
        if maximum_duration < MIN_DURATION:
            raise ProposalError(
                f"max_duration_sec must be at least {MIN_DURATION} seconds"
            )
        if start < 0:
            reasons.append("manifest_source_start_negative")
        if duration < MIN_DURATION or duration > maximum_duration:
            reasons.append("manifest_duration_out_of_range")
        if (
            canonical_decimal(row.get("boundary_start_sec_text"))
            != canonical_decimal(start)
            or canonical_decimal(row.get("boundary_end_sec_text"))
            != canonical_decimal(end)
        ):
            reasons.append("manifest_boundary_text_mismatch")
        if _decimal(row.get("duration_sec"), "duration_sec") != duration:
            reasons.append("manifest_duration_mismatch")
        if row.get("source_duration_sec") is not None and end > _decimal(
            row.get("source_duration_sec"), "source_duration_sec"
        ) + Decimal("0.001"):
            reasons.append("manifest_end_exceeds_source_duration")
        phase_values = {
            name: _decimal(row.get(name), name)
            for name in PHASE_FIELDS
        }
        _validate_phase_values(phase_values, duration)
        source_sha = str(row.get("source_sha256", ""))
        pipeline_version = str(row.get("pipeline_version", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha) or not pipeline_version:
            raise ProposalError("missing source hash or pipeline version")
        expected_signature = boundary_signature(
            source_sha,
            start,
            end,
            pipeline_version,
        )
        if row.get("boundary_signature") != expected_signature:
            reasons.append("manifest_boundary_signature_mismatch")
        expected_phase_signature = phase_signature(
            {**row, "boundary_signature": expected_signature}
        )
        if row.get("phase_signature") != expected_phase_signature:
            reasons.append("manifest_phase_signature_mismatch")
    except (ProposalError, KeyError, TypeError, ValueError):
        reasons.append("manifest_invalid")
    return list(dict.fromkeys(reasons))


def validate_pipeline(
    rows: Iterable[dict[str, Any]],
    work_root: Path,
    *,
    require_hand_keypoints: bool = False,
    verify_source_sha: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    source_root: Path | None = None,
) -> dict[str, Any]:
    work_root = Path(work_root)
    materialized = list(rows)
    errors: list[dict[str, str]] = []
    bad_ids: set[str] = set()
    source_hashes: dict[Path, str] = {}
    source_probes: dict[Path, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in materialized:
        clip_id = str(row.get("clip_id", ""))
        if clip_id in seen:
            errors.append({"clip_id": clip_id, "reason": "duplicate_clip_id"})
            bad_ids.add(clip_id)
        seen.add(clip_id)
        for reason in _manifest_row_errors(row):
            errors.append({"clip_id": clip_id, "reason": reason})
            bad_ids.add(clip_id)

        try:
            source, _ = resolve_source_video(row, source_root)
        except ProposalError:
            errors.append({"clip_id": clip_id, "reason": "missing_source"})
            bad_ids.add(clip_id)
            source = None
        if source is not None and not source.is_file():
            errors.append({"clip_id": clip_id, "reason": "missing_source"})
            bad_ids.add(clip_id)
        elif source is not None:
            if verify_source_sha:
                actual_source_sha = _cached_sha256(source_hashes, source)
                if actual_source_sha != row.get("source_sha256"):
                    errors.append({"clip_id": clip_id, "reason": "source_sha256_mismatch"})
                    bad_ids.add(clip_id)
            try:
                if source not in source_probes:
                    source_probes[source] = probe_video(source, ffprobe_bin)
                source_duration = float(source_probes[source]["duration_sec"])
                if float(row.get("source_end_sec")) > source_duration + 0.001:
                    errors.append(
                        {"clip_id": clip_id, "reason": "source_end_exceeds_duration"}
                    )
                    bad_ids.add(clip_id)
            except Exception:
                errors.append({"clip_id": clip_id, "reason": "source_probe_failed"})
                bad_ids.add(clip_id)

        output = Path(
            str(row.get("output_video") or work_root / "clips" / f"{clip_id}.mp4")
        )
        if row.get("export_status") != "ok" or not output.is_file():
            errors.append({"clip_id": clip_id, "reason": "missing_or_failed_export"})
            bad_ids.add(clip_id)
        else:
            actual_output_sha = sha256_file(output)
            if actual_output_sha != row.get("output_sha256"):
                errors.append({"clip_id": clip_id, "reason": "output_sha256_mismatch"})
                bad_ids.add(clip_id)
            try:
                info = probe_video(output, ffprobe_bin)
                verify_full_decode(output, ffmpeg_bin)
                info["full_decode_verified"] = True
                expected_width, expected_height = _target_dimensions(row, info)
                checks = _export_checks(
                    info,
                    float(row["duration_sec"]),
                    expected_width,
                    expected_height,
                )
                for name, valid in checks.items():
                    if not valid:
                        errors.append(
                            {"clip_id": clip_id, "reason": f"output_{name}_invalid"}
                        )
                        bad_ids.add(clip_id)
            except Exception:
                errors.append({"clip_id": clip_id, "reason": "output_probe_failed"})
                bad_ids.add(clip_id)

        if require_hand_keypoints:
            hand_path = work_root / "hand_keypoints" / f"{clip_id}_hands.json"
            payload = _load_json(hand_path)
            reasons = (
                ["keypoint_missing_output"]
                if payload is None
                else validate_keypoint_payload(
                    payload, row, require_provenance=True
                )
            )
            for reason in reasons:
                errors.append({"clip_id": clip_id, "reason": reason})
                bad_ids.add(clip_id)

    report = {
        "manifest_count": len(materialized),
        "unique_clip_id_count": len(seen),
        "valid_count": sum(
            str(row.get("clip_id", "")) not in bad_ids for row in materialized
        ),
        "error_count": len(errors),
        "hand_keypoints_required": require_hand_keypoints,
        "source_sha_verified": verify_source_sha,
        "review_status_counts": dict(
            Counter(str(row.get("review_status", "")) for row in materialized)
        ),
        "boundary_policy_sha256_values": sorted(
            {
                str(row["boundary_policy_sha256"])
                for row in materialized
                if row.get("boundary_policy_sha256")
            }
        ),
        "approved_trial_ids": sorted(
            {
                str(trial_id)
                for row in materialized
                for trial_id in row.get("approved_trial_ids", [])
            }
        ),
        "errors": errors,
    }
    report_path = work_root / "reports" / "validation.json"
    _atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2))
    return report


def prepare_manifest(
    proposals_path: Path,
    work_root: Path,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    *,
    source_root: Path | None = None,
    max_duration: Any = MAX_DURATION,
) -> list[dict[str, Any]]:
    rows = normalize_proposals(
        load_jsonl(Path(proposals_path)),
        pipeline_version,
        source_root=source_root,
        max_duration=max_duration,
    )
    manifest = Path(work_root) / "manifests" / "causal_segments.jsonl"
    write_jsonl(manifest, rows)
    summary = {
        "proposal_count": len(rows),
        "pipeline_version": pipeline_version,
        "duration_min_sec": min((row["duration_sec"] for row in rows), default=0.0),
        "duration_max_sec": max((row["duration_sec"] for row in rows), default=0.0),
        "review_status_counts": dict(Counter(row["review_status"] for row in rows)),
    }
    _atomic_write_text(
        Path(work_root) / "reports" / "prepare.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    return rows


def require_approved_boundary_policy(
    boundary_policy: Path | dict[str, Any] | None,
) -> dict[str, Any]:
    if boundary_policy is None:
        raise ProposalError(
            "full run requires --boundary-policy with status=approved"
        )
    if isinstance(boundary_policy, dict):
        policy = dict(boundary_policy)
        policy_source = None
        policy_sha = hashlib.sha256(
            json.dumps(policy, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    else:
        policy_path = Path(boundary_policy)
        if not policy_path.is_file():
            raise ProposalError(f"missing boundary policy: {policy_path}")
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ProposalError(f"invalid boundary policy JSON: {exc}") from exc
        if not isinstance(policy, dict):
            raise ProposalError("boundary policy must be a JSON object")
        policy_source = str(policy_path.resolve())
        policy_sha = sha256_file(policy_path)
    if policy.get("status") != "approved":
        raise ProposalError("boundary policy must have status=approved")
    approved_trial_ids = policy.get("approved_trial_ids")
    if (
        not isinstance(approved_trial_ids, list)
        or not approved_trial_ids
        or any(not isinstance(item, str) or not item.strip() for item in approved_trial_ids)
    ):
        raise ProposalError(
            "boundary policy approved_trial_ids must be a non-empty string list"
        )
    return {
        **policy,
        "approved_trial_ids": list(dict.fromkeys(approved_trial_ids)),
        "policy_source": policy_source,
        "policy_sha256": policy_sha,
    }


def run_pipeline(
    proposals_path: Path,
    work_root: Path,
    *,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    boundary_policy: Path | dict[str, Any] | None = None,
    run_hand_detection: bool = False,
    hand_model: Path | None = None,
    hand_detector: Path | None = None,
    python_executable: str = sys.executable,
    source_root: Path | None = None,
    max_duration: Any = MAX_DURATION,
) -> dict[str, Any]:
    work_root = Path(work_root)
    approved_policy = require_approved_boundary_policy(boundary_policy)
    _atomic_write_text(
        work_root / "reports" / "boundary_policy.json",
        json.dumps(approved_policy, ensure_ascii=False, indent=2),
    )
    normalized = prepare_manifest(
        proposals_path,
        work_root,
        pipeline_version,
        source_root=source_root,
        max_duration=max_duration,
    )
    for row in normalized:
        row.update(
            {
                "boundary_policy_status": approved_policy["status"],
                "boundary_policy_sha256": approved_policy["policy_sha256"],
                "approved_trial_ids": approved_policy["approved_trial_ids"],
            }
        )
    write_jsonl(
        work_root / "manifests" / "causal_segments.jsonl", normalized
    )
    exported = export_rows(normalized, work_root, source_root=source_root)
    if run_hand_detection:
        if hand_model is None:
            raise ProposalError("--hand-model is required with --run-hand-detection")
        detector = hand_detector or Path(__file__).with_name(
            "run_capture_hand_detection.py"
        )
        run_hand_detection_rows(
            exported,
            work_root,
            detector,
            hand_model,
            python_executable=python_executable,
        )
    return validate_pipeline(
        exported,
        work_root,
        require_hand_keypoints=run_hand_detection,
        verify_source_sha=True,
        source_root=source_root,
    )


def _smoke_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= limit <= MAX_SMOKE_EXPORTS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_SMOKE_EXPORTS}"
        )
    return limit


def _maximum_duration(value: str) -> Decimal:
    try:
        maximum = _decimal(value, "--max-duration")
    except ProposalError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if maximum < MIN_DURATION:
        raise argparse.ArgumentTypeError(
            f"must be at least {MIN_DURATION} seconds"
        )
    return maximum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="normalize proposal JSONL")
    prepare.add_argument("--proposals", required=True)
    prepare.add_argument("--work-root", required=True)
    prepare.add_argument("--pipeline-version", default=DEFAULT_PIPELINE_VERSION)
    prepare.add_argument("--source-root")
    prepare.add_argument(
        "--max-duration",
        type=_maximum_duration,
        default=MAX_DURATION,
    )

    export = subparsers.add_parser("export", help="export dynamic event clips")
    export.add_argument("--manifest", required=True)
    export.add_argument("--work-root", required=True)
    export.add_argument("--crf", type=int, default=20)
    export.add_argument("--source-root")
    export.add_argument(
        "--smoke-limit",
        type=_smoke_limit,
        required=True,
        help=f"export only the first N rows for smoke testing (1-{MAX_SMOKE_EXPORTS})",
    )

    validate = subparsers.add_parser("validate", help="validate clips and keypoints")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--work-root", required=True)
    validate.add_argument("--require-hand-keypoints", action="store_true")
    validate.add_argument("--skip-source-sha", action="store_true")
    validate.add_argument("--source-root")

    run = subparsers.add_parser("run", help="prepare, export, optionally detect, validate")
    run.add_argument("--proposals", required=True)
    run.add_argument("--work-root", required=True)
    run.add_argument("--pipeline-version", default=DEFAULT_PIPELINE_VERSION)
    run.add_argument("--boundary-policy", required=True)
    run.add_argument("--source-root")
    run.add_argument(
        "--max-duration",
        type=_maximum_duration,
        default=MAX_DURATION,
    )
    run.add_argument("--run-hand-detection", action="store_true")
    run.add_argument("--hand-model")
    run.add_argument(
        "--hand-detector",
        default=str(Path(__file__).with_name("run_capture_hand_detection.py")),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        rows = prepare_manifest(
            Path(args.proposals),
            Path(args.work_root),
            args.pipeline_version,
            source_root=Path(args.source_root) if args.source_root else None,
            max_duration=args.max_duration,
        )
        print(json.dumps({"prepared_count": len(rows)}, ensure_ascii=False))
        return 0
    if args.command == "export":
        manifest_rows = load_jsonl(Path(args.manifest))[: args.smoke_limit]
        manifest_rows = [
            {
                **row,
                "execution_scope": "smoke",
                "smoke_limit": args.smoke_limit,
            }
            for row in manifest_rows
        ]
        rows = export_rows(
            manifest_rows,
            Path(args.work_root),
            crf=args.crf,
            source_root=Path(args.source_root) if args.source_root else None,
        )
        error_count = sum(row.get("export_status") != "ok" for row in rows)
        print(
            json.dumps(
                {"input_count": len(rows), "error_count": error_count},
                ensure_ascii=False,
            )
        )
        return 0 if error_count == 0 else 2
    if args.command == "validate":
        report = validate_pipeline(
            load_jsonl(Path(args.manifest)),
            Path(args.work_root),
            require_hand_keypoints=args.require_hand_keypoints,
            verify_source_sha=not args.skip_source_sha,
            source_root=Path(args.source_root) if args.source_root else None,
        )
        print(
            json.dumps(
                {
                    "manifest_count": report["manifest_count"],
                    "valid_count": report["valid_count"],
                    "error_count": report["error_count"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["error_count"] == 0 else 2
    if args.command == "run":
        report = run_pipeline(
            Path(args.proposals),
            Path(args.work_root),
            pipeline_version=args.pipeline_version,
            boundary_policy=Path(args.boundary_policy),
            run_hand_detection=args.run_hand_detection,
            hand_model=Path(args.hand_model) if args.hand_model else None,
            hand_detector=Path(args.hand_detector),
            source_root=Path(args.source_root) if args.source_root else None,
            max_duration=args.max_duration,
        )
        print(
            json.dumps(
                {
                    "manifest_count": report["manifest_count"],
                    "valid_count": report["valid_count"],
                    "error_count": report["error_count"],
                    "hand_keypoints_required": report["hand_keypoints_required"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["error_count"] == 0 else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
