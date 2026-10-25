"""Fail-closed human-face scan for final video candidates.

This module only detects and records face evidence. It never approves privacy by
itself: a no-hit scan still requires visual review, and any detector/decode failure
remains unresolved.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Iterator, Optional

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent / ".cache" / "face_detection" / "face_detection_yunet_2023mar.onnx"


def _empty_report(status: str, *, error: Optional[str] = None) -> dict:
    return {
        "status": status,
        "detector_backend": "opencv_yunet",
        "detector_version": getattr(cv2, "__version__", None),
        "all_frames_checked": False,
        "frames_checked": 0,
        "frames_with_faces": 0,
        "face_count_detected": None,
        "hits": [],
        "error": error,
    }


def privacy_fields(report: dict) -> dict:
    status = report.get("status")
    if status in {"detector_failed", "decode_failed"} or not report.get("all_frames_checked"):
        return {
            "face_detection": report,
            "face_count_detected": None,
            "face_removal_method": None,
            "faces_remaining_after_crop": None,
            "subject_clarity_after_crop": None,
            "face_crop_required": None,
            "privacy_reviewed": False,
        }
    count = int(report.get("face_count_detected") or 0)
    return {
        "face_detection": report,
        "face_count_detected": count,
        "face_removal_method": "none_detected_pending_visual_review" if count == 0 else "fixed_crop_or_reject_required",
        "faces_remaining_after_crop": None,
        "subject_clarity_after_crop": None,
        "face_crop_required": count > 0,
        "privacy_reviewed": False,
    }


def create_detector(model: Path, width: int, height: int, *, score_threshold: float = 0.6):
    if cv2 is None:
        raise RuntimeError("opencv-python is unavailable")
    if not model.is_file():
        raise FileNotFoundError(f"YuNet model not found: {model}")
    return cv2.FaceDetectorYN.create(
        str(model), "", (int(width), int(height)), score_threshold, 0.3, 5000
    )


def inspect_frames(frames: Iterable[np.ndarray], times: Iterable[float], detector, expected_frames: int) -> dict:
    report = _empty_report("ready")
    hits = []
    frame_iter = iter(frames)
    time_iter = iter(times)
    frame_index = 0
    while True:
        try:
            frame = next(frame_iter)
        except StopIteration:
            break
        except Exception as error:
            report["status"] = "decode_failed"
            report["error"] = str(error)
            return report
        try:
            time_s = float(next(time_iter))
        except StopIteration:
            report["status"] = "decode_failed"
            report["error"] = "fewer timestamps than decoded frames"
            return report
        except Exception as error:
            report["status"] = "decode_failed"
            report["error"] = str(error)
            return report
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            report["status"] = "decode_failed"
            report["error"] = f"invalid decoded frame at index {frame_index}"
            return report
        try:
            if hasattr(detector, "setInputSize"):
                detector.setInputSize((int(frame.shape[1]), int(frame.shape[0])))
            detection = detector.detect(frame)
            if isinstance(detection, tuple) and len(detection) == 2:
                _, faces = detection
            else:
                faces = detection
        except Exception as error:
            report["status"] = "detector_failed"
            report["error"] = str(error)
            return report
        if faces is not None:
            if isinstance(faces, list) and faces and isinstance(faces[0], dict):
                face_rows = [(item.get("bbox_xywh", [0, 0, 0, 0]), item.get("score", 0.0)) for item in faces]
            else:
                array = np.asarray(faces)
                if array.size == 0:
                    face_rows = []
                else:
                    array = array.reshape(-1, array.shape[-1] if array.ndim > 1 else array.size)
                    face_rows = [([float(v) for v in row[:4]], float(row[-1])) for row in array]
            for bbox, score in face_rows:
                x, y, w, h = [float(v) for v in bbox[:4]]
                hits.append({"frame_index": frame_index, "time_s": round(time_s, 6), "bbox_xywh": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)], "score": round(float(score), 6)})
        frame_index += 1
    report["frames_checked"] = frame_index
    if frame_index != int(expected_frames):
        report["status"] = "decode_failed"
        report["error"] = f"decoded {frame_index} frames; expected {expected_frames}"
        return report
    report["status"] = "no_faces_detected_visual_review_required" if not hits else "faces_detected_fixed_crop_or_reject"
    report["all_frames_checked"] = True
    report["frames_with_faces"] = len({h["frame_index"] for h in hits})
    report["face_count_detected"] = len(hits)
    report["hits"] = hits
    return report


def _raw_frames(path: Path, width: int, height: int, fps: float) -> tuple[Iterator[np.ndarray], subprocess.Popen]:
    frame_bytes = int(width) * int(height) * 3
    process = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-threads", "2", "-filter_threads", "2", "-i", str(path), "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    def frames():
        assert process.stdout is not None
        while True:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((int(height), int(width), 3))
    return frames(), process


def scan_video(path: Path, *, model: Path = DEFAULT_MODEL, fps: Optional[float] = None) -> dict:
    try:
        metadata = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)
        stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        width, height = int(stream["width"]), int(stream["height"])
        source_fps = eval(stream.get("r_frame_rate", "16/1"), {"__builtins__": {}}, {})
        scan_fps = float(fps or source_fps)
        frame_count = int(round(float(metadata["format"]["duration"]) * scan_fps))
        detector = create_detector(model, width, height)
        frames, process = _raw_frames(path, width, height, scan_fps)
        report = inspect_frames(frames, (i / scan_fps for i in range(frame_count)), detector, frame_count)
        if process.stdout: process.stdout.close()
        process.wait(timeout=60)
        if process.returncode != 0 and report.get("status") not in {"decode_failed", "detector_failed"}:
            report = _empty_report("decode_failed", error=(process.stderr.read() if process.stderr else b"").decode(errors="replace")[-1000:])
        report.update({"path": str(path), "scan_fps": scan_fps, "width": width, "height": height, "expected_frames": frame_count})
        return report
    except Exception as error:
        return _empty_report("detector_failed", error=str(error)) | {"path": str(path)}

