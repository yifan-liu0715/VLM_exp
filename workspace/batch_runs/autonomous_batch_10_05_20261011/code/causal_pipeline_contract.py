"""Pure contracts shared by the mechanical batch path and tests."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable


def add_uncovered_second_pass_intervals(duration_s: float, intervals: Iterable[tuple[float, float]], *, min_gap_s: float = 0.25):
    """Return proposals plus source gaps, without merging distinct proposals."""
    duration = max(0.0, float(duration_s))
    normalized = []
    for start, end in intervals:
        start = max(0.0, float(start)); end = min(duration, float(end))
        if end > start:
            normalized.append((start, end))
    normalized.sort()
    merged = []
    for start, end in normalized:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    output = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor >= float(min_gap_s):
            output.append((round(cursor, 6), round(start, 6), "uncovered_source_interval"))
        output.append((round(start, 6), round(end, 6), "hand_proposal"))
        cursor = max(cursor, end)
    if duration - cursor >= float(min_gap_s):
        output.append((round(cursor, 6), round(duration, 6), "uncovered_source_interval"))
    return output


def full_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_response_end(response_evidence_s):
    if isinstance(response_evidence_s, (int, float)):
        return float(response_evidence_s)
    if isinstance(response_evidence_s, str):
        values = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", response_evidence_s)]
        if values:
            return values[-1]
    return None

def derive_export_review_state(evidence, metadata, caption_verified=False):
    evidence = evidence or {}
    metadata = metadata or {}
    semantic_reviewed = bool(
        evidence.get("hand_transition_checked")
        and evidence.get("response_complete")
        and evidence.get("response_evidence_s") is not None
    )
    normalized_frames = int(metadata.get("normalized_frame_count") or metadata.get("decoded_frames") or round(float(metadata.get("duration", 0.0)) * 16.0))
    stage1_window_ok = normalized_frames >= 49
    stage2_window_ok = normalized_frames >= 85
    full_stage2_window_ok = normalized_frames >= 90
    semantic_valid_short = semantic_reviewed and not full_stage2_window_ok
    state = "semantic_valid_short" if semantic_valid_short else "review_needed"
    return {
        "state": state, "status": state, "confidence": "review_needed",
        "semantic_reviewed": semantic_reviewed, "privacy_reviewed": False,
        "encoded_format_verified": False, "encoded_semantics_verified": False,
        "semantic_valid_short": semantic_valid_short, "training_ready": False,
        "stage1_window_ok": stage1_window_ok, "stage2_window_ok": stage2_window_ok,
        "normalized_frame_count": normalized_frames, "caption_verified": bool(caption_verified),
        "second_pass_required": not semantic_reviewed, "second_pass_completed": bool(semantic_reviewed),
        "source_coverage_status": "decision_interval_only", "accepted_for_training": False,
        "keypoint_ready": False,
    }
