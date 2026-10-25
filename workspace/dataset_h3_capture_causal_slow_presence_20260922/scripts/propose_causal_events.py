#!/usr/bin/env python3
"""Propose high-recall causal event windows from capture videos.

This scanner is deliberately a *proposal* stage.  MediaPipe hand landmarks
provide temporal anchors (appearance and movement of a hand); ffmpeg provides
the sampled frames and a conservative global-motion proxy.  Neither signal can
prove that an animal/object responded, so every emitted row remains pending
manual review and carries ``feedback_status=unverified``.

Unlike the legacy selector, this module never creates a fixed five-second grid.
It joins nearby hand-activity intervals, emits dynamic 2--20 second windows,
and splits only genuinely long continuous intervals.  The output can be fed to
``causal_clip_pipeline.py prepare`` after review.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np


MIN_EVENT_SEC = 2.0
MAX_EVENT_SEC = 20.0
DEFAULT_SAMPLE_FPS = 4.0
DEFAULT_BRIDGE_GAP_SEC = 2.0
DEFAULT_MIN_ACTIVE_SEC = 0.35
DEFAULT_PRE_ROLL_SEC = 0.8
DEFAULT_POST_ROLL_SEC = 1.0
DEFAULT_CAMERA_THRESHOLD = 0.08
HAND_RELOCATION_DISTANCE = 0.25
FRAME_W = 320
FRAME_H = 240
PRESENCE_FRAME_W = 640
PRESENCE_FRAME_H = 360

# The high-recall pass is deliberately separate from the baseline pass.  Its
# wider activity net is useful for finding missed episodes in long recordings,
# while the bounded maximum and explicit continuation metadata prevent it from
# masquerading as a fixed-grid segmenter.
HIGH_RECALL_MAX_RUN_SEC = 12.0
HIGH_RECALL_BRIDGE_GAP_SEC = 1.5
HIGH_RECALL_MIN_ACTIVE_SEC = 0.15
HIGH_RECALL_QUIET_GAP_SEC = 0.75
HIGH_RECALL_THRESHOLD_SCALE = 0.5
HIGH_RECALL_THRESHOLD_FLOOR = 0.008
HIGH_RECALL_THRESHOLD_CEILING = 0.20
PRESENCE_RECALL_BRIDGE_GAP_SEC = 0.5


@dataclasses.dataclass(frozen=True)
class ActivitySample:
    """One sampled hand-motion observation.

    ``score`` is a normalized hand-motion/visibility signal.  ``camera_score``
    is kept separate so a reviewer can see when a candidate may be dominated by
    camera movement.  The scanner does not use either value as semantic proof.
    """

    time_sec: float
    score: float
    hand_count: int
    camera_score: float = 0.0
    hand_centers: tuple[tuple[float, float], ...] = ()


@dataclasses.dataclass(frozen=True)
class SourceRecord:
    source: Path
    source_rel: str
    source_name: str
    source_sha256: str
    duration_sec: float
    width: int
    height: int
    fps: float | None


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_source_rel(value: str) -> str:
    """Normalize a source-relative path without permitting traversal."""

    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"source_rel must be a safe relative path: {value!r}")
    return path.as_posix()


def _interval_overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    overlap = max(0.0, right - left)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union > 0 else 0.0


def deduplicate_runs(
    runs: Iterable[tuple[float, float]], overlap_ratio: float = 0.5
) -> list[tuple[float, float]]:
    """Drop duplicate/near-duplicate intervals while preserving time order.

    Runs are already generated in temporal order in normal use.  Sorting here
    makes the helper deterministic for tests and for resumed scans.  The first
    interval wins; semantic ranking is intentionally left to manual review.
    """

    ordered = sorted(
        (float(start), float(end))
        for start, end in runs
        if math.isfinite(float(start))
        and math.isfinite(float(end))
        and float(end) > float(start)
    )
    kept: list[tuple[float, float]] = []
    for interval in ordered:
        if any(_interval_overlap_ratio(interval, previous) > overlap_ratio for previous in kept):
            continue
        kept.append(interval)
    return kept


def _split_run(start: float, end: float, max_run_sec: float) -> list[tuple[float, float]]:
    if end <= start:
        return []
    if max_run_sec <= 0:
        raise ValueError("max_run_sec must be positive")
    pieces: list[tuple[float, float]] = []
    cursor = float(start)
    while end - cursor > max_run_sec + 1e-9:
        pieces.append((cursor, cursor + max_run_sec))
        cursor += max_run_sec
    if end - cursor > 1e-9:
        pieces.append((cursor, float(end)))
    return pieces


def build_activity_runs(
    samples: Sequence[ActivitySample],
    *,
    threshold: float,
    bridge_gap_sec: float = DEFAULT_BRIDGE_GAP_SEC,
    min_active_sec: float = DEFAULT_MIN_ACTIVE_SEC,
    max_run_sec: float = MAX_EVENT_SEC,
) -> list[tuple[float, float]]:
    """Turn hand activity samples into merged temporal runs.

    A short inactive gap is bridged if the hand remains visible in the same
    image region. A substantial relocation across that gap is a separate
    review candidate, not proof of a different target. Long runs are split
    solely to respect the 20-second pipeline limit.
    """

    if bridge_gap_sec < 0 or min_active_sec < 0:
        raise ValueError("bridge_gap_sec and min_active_sec must be non-negative")
    ordered = sorted(samples, key=lambda sample: float(sample.time_sec))
    active_samples = [
        (index, sample)
        for index, sample in enumerate(ordered)
        if _finite(sample.score) >= float(threshold) and int(sample.hand_count) > 0
    ]
    if not active_samples:
        return []

    merged: list[tuple[float, float]] = []
    previous_index, first_sample = active_samples[0]
    start = previous = float(first_sample.time_sec)
    previous_sample = first_sample
    for current_index, active_sample in active_samples[1:]:
        current = float(active_sample.time_sec)
        gap_samples = ordered[previous_index + 1 : current_index]
        # A hand that remains visible through a short detector lull is likely
        # one continuous contact.  If it disappears, keep the events separate
        # even when their timestamps happen to be close.
        hand_visible_through_gap = all(
            int(sample.hand_count) > 0 for sample in gap_samples
        )
        previous_centers = getattr(previous_sample, "hand_centers", ())
        current_centers = getattr(active_sample, "hand_centers", ())
        relocated = bool(
            gap_samples
            and previous_centers
            and current_centers
            and _matched_displacement(previous_centers, current_centers)
            >= HAND_RELOCATION_DISTANCE
        )
        if (
            current - previous <= bridge_gap_sec + 1e-9
            and hand_visible_through_gap
            and not relocated
        ):
            previous = current
            previous_index, previous_sample = current_index, active_sample
            continue
        if previous - start >= min_active_sec - 1e-9:
            merged.append((start, previous))
        start = previous = current
        previous_index, previous_sample = current_index, active_sample
    if previous - start >= min_active_sec - 1e-9:
        merged.append((start, previous))

    pieces: list[tuple[float, float]] = []
    for start, end in merged:
        pieces.extend(_split_run(start, end, max_run_sec))
    return deduplicate_runs(pieces, overlap_ratio=0.99)


def build_hand_presence_runs(
    samples: Sequence[ActivitySample],
    *,
    bridge_gap_sec: float = PRESENCE_RECALL_BRIDGE_GAP_SEC,
) -> list[tuple[float, float]]:
    """Return natural hand-visible coverage intervals, independent of motion.

    Any sample with at least one detected hand is an anchor, including a hand
    whose motion score is zero. Nearby anchors are joined to tolerate a short
    detector dropout. The intervals are never divided by a duration grid;
    long intervals remain navigation coverage and require manual action
    segmentation.
    """

    bridge_gap_sec = float(bridge_gap_sec)
    if not math.isfinite(bridge_gap_sec) or bridge_gap_sec < 0:
        raise ValueError("bridge_gap_sec must be finite and non-negative")
    ordered = sorted(
        (
            sample
            for sample in samples
            if math.isfinite(float(sample.time_sec))
        ),
        key=lambda sample: float(sample.time_sec),
    )
    visible_anchors = [
        (index, sample)
        for index, sample in enumerate(ordered)
        if int(sample.hand_count) > 0
    ]
    if not visible_anchors:
        return []

    runs: list[tuple[float, float]] = []
    previous_index, first_sample = visible_anchors[0]
    start = previous = float(first_sample.time_sec)
    for current_index, current_sample in visible_anchors[1:]:
        current = float(current_sample.time_sec)
        dropout_samples = ordered[previous_index + 1 : current_index]
        if current - previous <= bridge_gap_sec + 1e-9:
            previous = current
            previous_index = current_index
            continue
        runs.append((start, previous))
        start = previous = current
        previous_index = current_index
    runs.append((start, previous))
    return runs


def adaptive_threshold(samples: Sequence[ActivitySample]) -> float:
    """Return a robust threshold for a hand-motion score series."""

    values = np.asarray([max(0.0, _finite(sample.score)) for sample in samples], dtype=np.float32)
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # Keep a low floor for slow gestures while rejecting detector jitter.
    return max(0.015, min(0.95, median + 1.75 * mad))


def high_recall_threshold(samples: Sequence[ActivitySample]) -> float:
    """Return a lower, bounded threshold for the optional recall pass.

    The baseline threshold is intentionally conservative.  A second pass is
    allowed to lower it so slow hand movements and partially tracked hands are
    surfaced for review.  This function does not make a semantic decision and
    callers must keep the resulting rows pending review.
    """

    baseline = adaptive_threshold(samples)
    return max(
        HIGH_RECALL_THRESHOLD_FLOOR,
        min(HIGH_RECALL_THRESHOLD_CEILING, baseline * HIGH_RECALL_THRESHOLD_SCALE),
    )


def _merge_short_runs(
    runs: Sequence[tuple[float, float]], minimum_sec: float
) -> list[tuple[float, float]]:
    """Merge tiny adjacent pieces created by a quiet-valley split."""

    if not runs:
        return []
    minimum_sec = max(0.0, float(minimum_sec))
    merged: list[tuple[float, float]] = []
    for start, end in runs:
        start, end = float(start), float(end)
        if end <= start:
            continue
        if merged and end - start < minimum_sec:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < minimum_sec:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged


def build_high_recall_runs(
    samples: Sequence[ActivitySample],
    *,
    threshold: float | None = None,
    bridge_gap_sec: float = HIGH_RECALL_BRIDGE_GAP_SEC,
    min_active_sec: float = HIGH_RECALL_MIN_ACTIVE_SEC,
    max_run_sec: float = HIGH_RECALL_MAX_RUN_SEC,
    quiet_gap_sec: float = HIGH_RECALL_QUIET_GAP_SEC,
    quiet_threshold: float | None = None,
) -> list[tuple[float, float]]:
    """Build high-recall, event-shaped runs for a second review pass.

    First, a lower hand-motion threshold finds a broad set of possible
    activity.  Within each broad run, a sustained low-motion valley while the
    hand remains visible becomes a candidate boundary.  Finally, genuinely
    long continuous runs are bounded by ``max_run_sec`` and marked by the
    caller as continuations.  No fixed five-second grid is introduced.
    """

    if quiet_gap_sec < 0:
        raise ValueError("quiet_gap_sec must be non-negative")
    effective_threshold = (
        high_recall_threshold(samples) if threshold is None else float(threshold)
    )
    if not math.isfinite(effective_threshold) or effective_threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if quiet_threshold is None:
        quiet_threshold = max(
            HIGH_RECALL_THRESHOLD_FLOOR / 2.0,
            min(0.04, effective_threshold * 0.65),
        )
    quiet_threshold = float(quiet_threshold)
    if not math.isfinite(quiet_threshold) or quiet_threshold < 0:
        raise ValueError("quiet_threshold must be finite and non-negative")

    broad_runs = build_activity_runs(
        samples,
        threshold=effective_threshold,
        bridge_gap_sec=bridge_gap_sec,
        min_active_sec=min_active_sec,
        max_run_sec=max_run_sec,
    )
    if not broad_runs:
        return []

    ordered = sorted(samples, key=lambda sample: float(sample.time_sec))
    output: list[tuple[float, float]] = []
    for broad_start, broad_end in broad_runs:
        anchors = [
            (index, sample)
            for index, sample in enumerate(ordered)
            if broad_start - 1e-9 <= float(sample.time_sec) <= broad_end + 1e-9
            and int(sample.hand_count) > 0
            and _finite(sample.score) >= effective_threshold
        ]
        if len(anchors) < 2:
            output.extend(_split_run(broad_start, broad_end, max_run_sec))
            continue

        boundaries: list[float] = []
        previous_index, previous_sample = anchors[0]
        for current_index, current_sample in anchors[1:]:
            gap_samples = ordered[previous_index + 1 : current_index]
            gap_duration = float(current_sample.time_sec) - float(previous_sample.time_sec)
            quiet_valley = bool(gap_samples) and all(
                int(sample.hand_count) > 0
                and _finite(sample.score) < quiet_threshold
                for sample in gap_samples
            )
            if quiet_valley and gap_duration >= float(quiet_gap_sec) - 1e-9:
                # Place the boundary in the middle of the observed quiet
                # valley.  This leaves a small amount of context on either
                # side without inventing an onset or response timestamp.
                boundaries.append(
                    (float(previous_sample.time_sec) + float(current_sample.time_sec))
                    / 2.0
                )
            previous_index, previous_sample = current_index, current_sample

        edges = [float(broad_start), *boundaries, float(broad_end)]
        pieces = [
            (left, right)
            for left, right in zip(edges, edges[1:])
            if right - left >= 1e-9
        ]
        pieces = _merge_short_runs(pieces, max(float(min_active_sec), 0.5))
        for piece in pieces:
            output.extend(_split_run(piece[0], piece[1], max_run_sec))

    return deduplicate_runs(output, overlap_ratio=0.99)


def _expand_to_min_duration(
    start: float, end: float, source_duration: float, minimum: float = MIN_EVENT_SEC
) -> tuple[float, float]:
    if source_duration < minimum - 1e-9:
        raise ValueError("source is shorter than the minimum event duration")
    start = max(0.0, min(float(start), source_duration))
    end = max(start, min(float(end), source_duration))
    if end - start >= minimum - 1e-9:
        return start, end
    needed = minimum - (end - start)
    left = min(start, needed / 2.0)
    start -= left
    needed -= left
    right = min(source_duration - end, needed)
    end += right
    needed -= right
    if needed > 1e-9:
        left = min(start, needed)
        start -= left
        needed -= left
    if needed > 1e-9:
        end = min(source_duration, end + needed)
    return start, end


def build_proposal(
    *,
    source: Path,
    source_rel: str,
    source_sha256: str,
    source_duration_sec: float,
    width: int,
    height: int,
    fps: float | None,
    run: tuple[float, float],
    candidate_index: int,
    threshold: float,
    camera_motion_score: float = 0.0,
    pre_roll_sec: float = DEFAULT_PRE_ROLL_SEC,
    post_roll_sec: float = DEFAULT_POST_ROLL_SEC,
    motion_score: float | None = None,
    hand_visible_rate: float | None = None,
    continuation_group: str | None = None,
    continuation_index: int | None = None,
    continuation_count: int | None = None,
    clip_start_floor: float | None = None,
    candidate_method: str = "hand_activity_anchor",
    proposal_origin: str = "auto_hand_anchor_v1",
    clip_id_prefix: str = "causal",
    max_clip_duration_sec: float | None = MAX_EVENT_SEC,
    minimum_clip_duration_sec: float = MIN_EVENT_SEC,
) -> dict[str, Any]:
    """Build one reviewable causal proposal from a hand-activity interval."""

    duration = max(0.0, float(source_duration_sec))
    onset, offset = map(float, run)
    onset = max(0.0, min(onset, duration))
    offset = max(onset, min(offset, duration))
    pre = min(onset, max(0.0, float(pre_roll_sec)))
    post = min(duration - offset, max(0.0, float(post_roll_sec)))
    if clip_start_floor is not None:
        pre = min(pre, max(0.0, onset - float(clip_start_floor)))
    if max_clip_duration_sec is not None:
        max_clip_duration_sec = float(max_clip_duration_sec)
        if not math.isfinite(max_clip_duration_sec) or max_clip_duration_sec <= 0:
            raise ValueError("max_clip_duration_sec must be finite and positive")
        context_budget = max(0.0, max_clip_duration_sec - (offset - onset))
        if pre + post > context_budget:
            scale = context_budget / (pre + post)
            pre *= scale
            post *= scale
    raw_start = onset - pre
    if clip_start_floor is not None:
        raw_start = max(raw_start, min(float(clip_start_floor), duration))
    raw_end = offset + post
    minimum_clip_duration_sec = float(minimum_clip_duration_sec)
    if not math.isfinite(minimum_clip_duration_sec) or minimum_clip_duration_sec < 0:
        raise ValueError("minimum_clip_duration_sec must be finite and non-negative")
    clip_start, clip_end = _expand_to_min_duration(
        raw_start,
        raw_end,
        duration,
        minimum=minimum_clip_duration_sec,
    )
    clip_duration = clip_end - clip_start
    if (
        max_clip_duration_sec is not None
        and clip_duration > max_clip_duration_sec + 1e-9
    ):
        # This should only occur when callers pass an unsplit run.
        clip_end = min(duration, clip_start + max_clip_duration_sec)
        clip_duration = clip_end - clip_start

    action_start = max(0.0, min(clip_duration, onset - clip_start))
    action_end = max(action_start, min(clip_duration, offset - clip_start))
    if action_end - action_start < 1e-6:
        # A one-sample anchor is still useful to a reviewer.  Give it a tiny,
        # deterministic phase rather than claiming a zero-length action.
        action_end = min(clip_duration, action_start + min(0.25, clip_duration))
    response_start = action_end
    response_end = min(clip_duration, response_start + min(1.0, clip_duration - response_start))
    post_start = response_end
    post_end = clip_duration

    camera_score = max(0.0, _finite(camera_motion_score))
    camera_motion = "possible_camera_motion" if camera_score >= DEFAULT_CAMERA_THRESHOLD else "low_camera_motion"
    stem = source.stem.replace(" ", "_")
    row: dict[str, Any] = {
        "clip_id": f"{clip_id_prefix}_{stem}_{int(candidate_index):05d}",
        "source_video": str(source),
        "source_rel": safe_source_rel(source_rel),
        "source_name": source.name,
        "source_sha256": str(source_sha256),
        "source_duration_sec": round(duration, 6),
        "source_start_sec": round(clip_start, 6),
        "source_end_sec": round(clip_end, 6),
        "duration_sec": round(clip_duration, 6),
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "candidate_method": candidate_method,
        "proposal_origin": proposal_origin,
        "review_status": "pending_user_review",
        "quality": "auto_candidate",
        "reject_reason": None,
        "camera_motion": camera_motion,
        "camera_motion_score": round(camera_score, 6),
        "motion_score": None if motion_score is None else round(float(motion_score), 6),
        "hand_motion_threshold": round(float(threshold), 6),
        "hand_visible_rate": None if hand_visible_rate is None else round(float(hand_visible_rate), 6),
        "hand_onset_sec_abs": round(onset, 6),
        "hand_offset_sec_abs": round(offset, 6),
        "pre_start_sec_rel": 0.0,
        "pre_end_sec_rel": round(action_start, 6),
        "action_start_sec_rel": round(action_start, 6),
        "action_end_sec_rel": round(action_end, 6),
        "response_start_sec_rel": round(response_start, 6),
        "response_end_sec_rel": round(response_end, 6),
        "post_start_sec_rel": round(post_start, 6),
        "post_end_sec_rel": round(post_end, 6),
        # The scanner cannot see an animal/object state transition reliably.
        "semantic_feedback_verified": False,
        "semantic_action_verified": False,
        "semantic_action_status": "unverified",
        "feedback_status": "unverified",
        "feedback_evidence": "hand_motion_only_pending_visual_review",
        "phase_source": "hand_motion_anchor_only",
        "continuous_contact_possible": True,
    }
    if continuation_group:
        row.update(
            {
                "continuation_group": continuation_group,
                "continuation_index": continuation_index,
                "continuation_count": continuation_count,
            }
        )
    return row


def build_hand_presence_proposal(
    *,
    presence_detected: bool = True,
    fallback_type: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a coverage/navigation row without asserting an action boundary."""

    row = build_proposal(
        max_clip_duration_sec=None,
        minimum_clip_duration_sec=0.0,
        **kwargs,
    )
    fallback_type = None if presence_detected else (fallback_type or "zero_hand_detection")
    is_exclusion = fallback_type == "empty_decode_trace"
    row.update(
        {
            "quality": (
                "coverage_candidate"
                if presence_detected
                else "source_exclusion" if is_exclusion else "coverage_fallback"
            ),
            "candidate_scope": (
                "coverage_navigation_not_semantic_action"
                if presence_detected
                else "source_exclusion" if is_exclusion else "whole_source_navigation_fallback"
            ),
            "navigation_only": True,
            "fallback_type": fallback_type,
            "hand_onset_sec_abs": (
                row["hand_onset_sec_abs"] if presence_detected else None
            ),
            "hand_offset_sec_abs": (
                row["hand_offset_sec_abs"] if presence_detected else None
            ),
            "hand_presence_start_sec_abs": (
                row["hand_onset_sec_abs"] if presence_detected else None
            ),
            "hand_presence_end_sec_abs": (
                row["hand_offset_sec_abs"] if presence_detected else None
            ),
            "hand_presence_signal": (
                "detected_hand_count_gt_zero"
                if presence_detected
                else (
                    "no_sampled_frames"
                    if is_exclusion
                    else "source_too_short_unscanned"
                    if fallback_type == "too_short_source"
                    else "no_hand_detected_whole_source_fallback"
                )
            ),
            "zero_hand_detection_fallback": fallback_type == "zero_hand_detection",
            "semantic_action_verified": False,
            "semantic_action_status": "unverified",
            "action_boundary_status": "unverified",
            "semantic_feedback_verified": False,
            "feedback_status": "unverified",
            "feedback_evidence": (
                "hand_presence_only_pending_visual_review"
                if presence_detected
                else "detector_miss_whole_source_pending_visual_review"
            ),
            "phase_source": (
                "hand_presence_bounds_only_no_semantic_action_boundary"
                if presence_detected
                else "empty_decode_no_action_boundary"
                if is_exclusion
                else "too_short_whole_source_no_action_boundary"
                if fallback_type == "too_short_source"
                else "zero_detection_whole_source_no_action_boundary"
            ),
            "continuous_contact_possible": bool(presence_detected),
            "requires_manual_action_segmentation": True,
            "export_eligible": False,
        }
    )
    return row


def build_source_presence_fallback(
    record: SourceRecord,
    *,
    candidate_index: int,
    fallback_type: str,
    pre_roll_sec: float = DEFAULT_PRE_ROLL_SEC,
    post_roll_sec: float = DEFAULT_POST_ROLL_SEC,
) -> dict[str, Any]:
    """Represent one source that lacks a normal detected-hand coverage run."""

    methods = {
        "zero_hand_detection": "whole_source_zero_hand_fallback",
        "too_short_source": "whole_source_too_short_fallback",
        "empty_decode_trace": "empty_decode_exclusion",
    }
    if fallback_type not in methods:
        raise ValueError(f"unsupported presence fallback type: {fallback_type}")
    return build_hand_presence_proposal(
        source=record.source,
        source_rel=record.source_rel,
        source_sha256=record.source_sha256,
        source_duration_sec=record.duration_sec,
        width=record.width,
        height=record.height,
        fps=record.fps,
        run=(0.0, float(record.duration_sec)),
        candidate_index=candidate_index,
        threshold=0.0,
        pre_roll_sec=pre_roll_sec,
        post_roll_sec=post_roll_sec,
        motion_score=0.0,
        hand_visible_rate=0.0,
        candidate_method=methods[fallback_type],
        proposal_origin="auto_hand_presence_recall_v1",
        clip_id_prefix="causal_hp",
        presence_detected=False,
        fallback_type=fallback_type,
    )


def _run_json(command: list[str]) -> dict[str, Any]:
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(process.stdout)


def probe_video(path: Path) -> dict[str, Any]:
    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = next(
        item for item in data.get("streams", []) if item.get("codec_type") == "video"
    )
    rate = str(stream.get("avg_frame_rate") or "0/1")
    try:
        numerator, denominator = rate.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = None
    return {
        "duration_sec": float(data.get("format", {}).get("duration") or 0.0),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
    }


def _iter_frames(
    path: Path,
    sample_fps: float,
    *,
    frame_width: int = FRAME_W,
    frame_height: int = FRAME_H,
):
    frame_width = int(frame_width)
    frame_height = int(frame_height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    frame_bytes = frame_width * frame_height * 3
    filter_graph = (
        f"fps={sample_fps},scale={frame_width}:{frame_height}:force_original_aspect_ratio=decrease,"
        f"pad={frame_width}:{frame_height}:(ow-iw)/2:(oh-ih)/2"
    )
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-an",
            "-vf",
            filter_graph,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    index = 0

    def read_exact(size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = process.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    try:
        while True:
            # A Windows pipe is allowed to return a short read even while the
            # producer is still alive; only treat an actually incomplete frame
            # at EOF as the end of the stream.
            chunk = read_exact(frame_bytes)
            if len(chunk) != frame_bytes:
                break
            yield index / sample_fps, np.frombuffer(chunk, dtype=np.uint8).reshape(
                frame_height, frame_width, 3
            )
            index += 1
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg decode failed ({code}): {stderr[-500:]}")


def _landmark_centers(result: Any) -> list[tuple[float, float]]:
    centers: list[tuple[float, float]] = []
    for landmarks in getattr(result, "hand_landmarks", []) or []:
        ids = (0, 5, 9, 13, 17)
        centers.append(
            (
                sum(float(landmarks[index].x) for index in ids) / len(ids),
                sum(float(landmarks[index].y) for index in ids) / len(ids),
            )
        )
    return centers


def _matched_displacement(
    previous: Sequence[tuple[float, float]], current: Sequence[tuple[float, float]]
) -> float:
    if not previous or not current:
        return 0.18 if current else 0.0
    remaining = list(previous)
    distances: list[float] = []
    for x, y in current:
        if not remaining:
            distances.append(0.18)
            continue
        index, nearest = min(
            enumerate(remaining),
            key=lambda item: (item[1][0] - x) ** 2 + (item[1][1] - y) ** 2,
        )
        remaining.pop(index)
        distances.append(math.hypot(nearest[0] - x, nearest[1] - y))
    return max(distances, default=0.0)


def scan_video(
    record: SourceRecord,
    *,
    model_path: Path,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    frame_width: int = FRAME_W,
    frame_height: int = FRAME_H,
) -> tuple[list[ActivitySample], dict[str, Any]]:
    """Run MediaPipe on sampled frames and return motion observations."""

    # Imports are lazy so pure manifest tests work in environments without the
    # optional MediaPipe runtime.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    samples: list[ActivitySample] = []
    previous_centers: list[tuple[float, float]] = []
    previous_gray: np.ndarray | None = None
    visible = 0
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for index, (time_sec, frame) in enumerate(
            _iter_frames(
                record.source,
                sample_fps,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        ):
            rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(image, int(round(time_sec * 1000.0)))
            centers = _landmark_centers(result)
            hand_count = len(centers)
            if hand_count:
                visible += 1
            hand_motion = _matched_displacement(previous_centers, centers)
            # Border difference is a cheap camera-motion warning.  It remains
            # a warning only; moving hands can also affect this score.
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            camera_score = 0.0
            if previous_gray is not None:
                border = np.concatenate(
                    [gray[:24].ravel(), gray[-24:].ravel(), gray[:, :32].ravel(), gray[:, -32:].ravel()]
                )
                previous_border = np.concatenate(
                    [previous_gray[:24].ravel(), previous_gray[-24:].ravel(), previous_gray[:, :32].ravel(), previous_gray[:, -32:].ravel()]
                )
                camera_score = float(np.mean(np.abs(border.astype(np.int16) - previous_border.astype(np.int16))) / 255.0)
            visibility_signal = 0.08 if hand_count and not previous_centers else 0.0
            samples.append(
                ActivitySample(
                    time_sec=float(time_sec),
                    score=float(min(1.0, hand_motion + visibility_signal)),
                    hand_count=hand_count,
                    camera_score=camera_score,
                    hand_centers=tuple(centers),
                )
            )
            previous_centers = centers
            previous_gray = gray
    return samples, {
        "sample_count": len(samples),
        "hand_visible_rate": visible / len(samples) if samples else 0.0,
        "mediapipe_version": getattr(mp, "__version__", "unknown"),
        "scan_frame_width": int(frame_width),
        "scan_frame_height": int(frame_height),
    }


def _record_from_inventory(row: dict[str, Any], source_root: Path | None) -> SourceRecord:
    raw_rel = str(row.get("source_rel") or "")
    rel = safe_source_rel(raw_rel) if raw_rel and not Path(raw_rel).is_absolute() and ":" not in raw_rel[:3] else Path(raw_rel).name
    if source_root is not None:
        source = (Path(source_root) / Path(rel)).resolve()
    else:
        source = Path(str(row.get("source_path") or row.get("source_video") or raw_rel)).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    probe = row.get("probe") if isinstance(row.get("probe"), dict) else row
    source_sha = str(row.get("sha256") or row.get("source_sha256") or "")
    if len(source_sha) != 64:
        source_sha = sha256_file(source)
    return SourceRecord(
        source=source,
        source_rel=rel,
        source_name=str(row.get("source_name") or source.name),
        source_sha256=source_sha,
        duration_sec=_finite(probe.get("duration_sec")),
        width=int(probe.get("width") or 0),
        height=int(probe.get("height") or 0),
        fps=_finite(probe.get("fps"), 0.0) or None,
    )


def load_inventory(path: Path, source_root: Path | None = None) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            records.append(_record_from_inventory(row, source_root))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return sorted(records, key=lambda item: item.source_rel.casefold())


def discover_inventory(source_root: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for source in sorted(
        (path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.relative_to(source_root).as_posix().casefold(),
    ):
        info = probe_video(source)
        records.append(
            SourceRecord(
                source=source,
                source_rel=source.relative_to(source_root).as_posix(),
                source_name=source.name,
                source_sha256=sha256_file(source),
                duration_sec=info["duration_sec"],
                width=info["width"],
                height=info["height"],
                fps=info["fps"],
            )
        )
    return records


def propose_for_samples(
    record: SourceRecord,
    samples: Sequence[ActivitySample],
    *,
    candidate_start: int,
    threshold: float | None = None,
    bridge_gap_sec: float = DEFAULT_BRIDGE_GAP_SEC,
    min_active_sec: float = DEFAULT_MIN_ACTIVE_SEC,
    pre_roll_sec: float = DEFAULT_PRE_ROLL_SEC,
    post_roll_sec: float = DEFAULT_POST_ROLL_SEC,
    mode: str = "standard",
    high_recall_max_run_sec: float = HIGH_RECALL_MAX_RUN_SEC,
    high_recall_bridge_gap_sec: float = HIGH_RECALL_BRIDGE_GAP_SEC,
    high_recall_min_active_sec: float = HIGH_RECALL_MIN_ACTIVE_SEC,
    high_recall_quiet_gap_sec: float = HIGH_RECALL_QUIET_GAP_SEC,
    presence_bridge_gap_sec: float = PRESENCE_RECALL_BRIDGE_GAP_SEC,
) -> list[dict[str, Any]]:
    if mode not in {"standard", "high_recall", "presence_recall"}:
        raise ValueError(f"unsupported proposal mode: {mode}")
    presence_detected = True
    presence_fallback_type: str | None = None
    if mode == "presence_recall":
        threshold = 0.0
        runs = build_hand_presence_runs(
            samples,
            bridge_gap_sec=presence_bridge_gap_sec,
        )
        if not runs and not samples:
            runs = [(0.0, float(record.duration_sec))]
            presence_detected = False
            presence_fallback_type = "empty_decode_trace"
        elif not runs and not any(int(sample.hand_count) > 0 for sample in samples):
            runs = [(0.0, float(record.duration_sec))]
            presence_detected = False
            presence_fallback_type = "zero_hand_detection"
        candidate_method = (
            "hand_presence_recall"
            if presence_detected
            else (
                "empty_decode_exclusion"
                if presence_fallback_type == "empty_decode_trace"
                else "whole_source_zero_hand_fallback"
            )
        )
        proposal_origin = "auto_hand_presence_recall_v1"
        clip_id_prefix = "causal_hp"
    elif mode == "high_recall":
        threshold = high_recall_threshold(samples) if threshold is None else float(threshold)
        runs = build_high_recall_runs(
            samples,
            threshold=threshold,
            bridge_gap_sec=high_recall_bridge_gap_sec,
            min_active_sec=high_recall_min_active_sec,
            max_run_sec=high_recall_max_run_sec,
            quiet_gap_sec=high_recall_quiet_gap_sec,
        )
        candidate_method = "hand_activity_high_recall"
        proposal_origin = "auto_hand_anchor_recall_v2"
        clip_id_prefix = "causal_hr"
    else:
        if threshold is None:
            threshold = adaptive_threshold(samples)
        runs = build_activity_runs(
            samples,
            threshold=threshold,
            bridge_gap_sec=bridge_gap_sec,
            min_active_sec=min_active_sec,
            max_run_sec=MAX_EVENT_SEC,
        )
        candidate_method = "hand_activity_anchor"
        proposal_origin = "auto_hand_anchor_v1"
        clip_id_prefix = "causal"
    if not runs:
        return []
    camera_scores = [sample.camera_score for sample in samples]
    camera_score = max(camera_scores, default=0.0)
    motion_score = max((sample.score for sample in samples), default=0.0)
    visible_rate = (
        sum(1 for sample in samples if sample.hand_count > 0) / len(samples)
        if samples
        else 0.0
    )
    continuation_groups: list[list[tuple[float, float]]] = []
    # Runs are already split at 20 seconds.  A group is created when adjacent
    # pieces are exactly contiguous; this marks continuation without claiming
    # that they are separate semantic actions.
    for run in runs:
        if continuation_groups and abs(run[0] - continuation_groups[-1][-1][1]) <= 1e-6:
            continuation_groups[-1].append(run)
        else:
            continuation_groups.append([run])

    output: list[dict[str, Any]] = []
    next_index = candidate_start
    previous_window_end: float | None = None
    for group_runs in continuation_groups:
        group_id = None
        if len(group_runs) > 1:
            group_id = f"{record.source.stem}:run:{next_index:05d}"
        # The recall pass can create adjacent event candidates separated by a
        # short quiet valley.  Carrying the floor across groups prevents the
        # contextual pre-rolls from turning distinct events into overlapping
        # clips, which would make the later merge stage discard useful rows.
        if mode == "standard":
            previous_window_end = None
        for piece_index, run in enumerate(group_runs, 1):
            proposal_builder = (
                build_hand_presence_proposal
                if mode == "presence_recall"
                else build_proposal
            )
            proposal_kwargs = dict(
                source=record.source,
                source_rel=record.source_rel,
                source_sha256=record.source_sha256,
                source_duration_sec=record.duration_sec,
                width=record.width,
                height=record.height,
                fps=record.fps,
                run=run,
                candidate_index=next_index,
                threshold=threshold,
                camera_motion_score=camera_score,
                pre_roll_sec=pre_roll_sec,
                post_roll_sec=post_roll_sec,
                motion_score=motion_score,
                hand_visible_rate=visible_rate,
                continuation_group=group_id,
                continuation_index=piece_index if group_id else None,
                continuation_count=len(group_runs) if group_id else None,
                clip_start_floor=previous_window_end,
                candidate_method=candidate_method,
                proposal_origin=proposal_origin,
                clip_id_prefix=clip_id_prefix,
            )
            if mode == "presence_recall":
                proposal_kwargs["presence_detected"] = presence_detected
                proposal_kwargs["fallback_type"] = presence_fallback_type
            row = proposal_builder(**proposal_kwargs)
            output.append(row)
            previous_window_end = float(row["source_end_sec"])
            next_index += 1
    return output


def propose_high_recall_for_samples(
    record: SourceRecord,
    samples: Sequence[ActivitySample],
    *,
    candidate_start: int,
    threshold: float | None = None,
    max_run_sec: float = HIGH_RECALL_MAX_RUN_SEC,
    bridge_gap_sec: float = HIGH_RECALL_BRIDGE_GAP_SEC,
    min_active_sec: float = HIGH_RECALL_MIN_ACTIVE_SEC,
    quiet_gap_sec: float = HIGH_RECALL_QUIET_GAP_SEC,
    pre_roll_sec: float = DEFAULT_PRE_ROLL_SEC,
    post_roll_sec: float = DEFAULT_POST_ROLL_SEC,
) -> list[dict[str, Any]]:
    """Convenience wrapper for the independent high-recall proposal pass."""

    return propose_for_samples(
        record,
        samples,
        candidate_start=candidate_start,
        threshold=threshold,
        pre_roll_sec=pre_roll_sec,
        post_roll_sec=post_roll_sec,
        mode="high_recall",
        high_recall_max_run_sec=max_run_sec,
        high_recall_bridge_gap_sec=bridge_gap_sec,
        high_recall_min_active_sec=min_active_sec,
        high_recall_quiet_gap_sec=quiet_gap_sec,
    )


def propose_presence_for_samples(
    record: SourceRecord,
    samples: Sequence[ActivitySample],
    *,
    candidate_start: int,
    bridge_gap_sec: float = PRESENCE_RECALL_BRIDGE_GAP_SEC,
    pre_roll_sec: float = DEFAULT_PRE_ROLL_SEC,
    post_roll_sec: float = DEFAULT_POST_ROLL_SEC,
) -> list[dict[str, Any]]:
    """Build hand-presence coverage rows for later manual action review."""

    return propose_for_samples(
        record,
        samples,
        candidate_start=candidate_start,
        pre_roll_sec=pre_roll_sec,
        post_roll_sec=post_roll_sec,
        mode="presence_recall",
        presence_bridge_gap_sec=bridge_gap_sec,
    )


def _scan_record(
    record: SourceRecord,
    model_path: Path,
    sample_fps: float,
    frame_width: int,
    frame_height: int,
    threshold: float | None,
    bridge_gap_sec: float,
    min_active_sec: float,
    pre_roll_sec: float,
    post_roll_sec: float,
    candidate_start: int,
    mode: str = "standard",
    high_recall_max_run_sec: float = HIGH_RECALL_MAX_RUN_SEC,
    high_recall_bridge_gap_sec: float = HIGH_RECALL_BRIDGE_GAP_SEC,
    high_recall_min_active_sec: float = HIGH_RECALL_MIN_ACTIVE_SEC,
    high_recall_quiet_gap_sec: float = HIGH_RECALL_QUIET_GAP_SEC,
    presence_bridge_gap_sec: float = PRESENCE_RECALL_BRIDGE_GAP_SEC,
    precomputed_scan: tuple[list[ActivitySample], dict[str, Any]] | None = None,
) -> tuple[SourceRecord, list[dict[str, Any]], dict[str, Any], str | None]:
    try:
        if record.duration_sec < MIN_EVENT_SEC:
            if mode == "presence_recall":
                fallback = build_source_presence_fallback(
                    record,
                    candidate_index=candidate_start,
                    fallback_type="too_short_source",
                    pre_roll_sec=pre_roll_sec,
                    post_roll_sec=post_roll_sec,
                )
                return (
                    record,
                    [fallback],
                    {
                        "status": "ok",
                        "source_status": "too_short",
                        "sample_count": 0,
                        "threshold": 0.0,
                        "candidate_count": 1,
                        "coverage_row_type": "too_short_source",
                        "detection_trace_cache_hit": False,
                        "_detection_trace": [],
                    },
                    None,
                )
            return record, [], {"status": "too_short", "sample_count": 0}, None
        if precomputed_scan is None:
            samples, metrics = scan_video(
                record,
                model_path=model_path,
                sample_fps=sample_fps,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            detection_trace_cache_hit = False
        else:
            samples, metrics = precomputed_scan
            detection_trace_cache_hit = True
        samples = list(samples)
        metrics = dict(metrics)
        if mode == "presence_recall":
            effective_threshold = 0.0
        elif mode == "high_recall":
            effective_threshold = high_recall_threshold(samples) if threshold is None else threshold
        else:
            effective_threshold = adaptive_threshold(samples) if threshold is None else threshold
        proposals = propose_for_samples(
            record,
            samples,
            candidate_start=candidate_start,
            threshold=effective_threshold,
            bridge_gap_sec=bridge_gap_sec,
            min_active_sec=min_active_sec,
            pre_roll_sec=pre_roll_sec,
            post_roll_sec=post_roll_sec,
            mode=mode,
            high_recall_max_run_sec=high_recall_max_run_sec,
            high_recall_bridge_gap_sec=high_recall_bridge_gap_sec,
            high_recall_min_active_sec=high_recall_min_active_sec,
            high_recall_quiet_gap_sec=high_recall_quiet_gap_sec,
            presence_bridge_gap_sec=presence_bridge_gap_sec,
        )
        metrics.update(
            {
                "status": "ok",
                "threshold": effective_threshold,
                "candidate_count": len(proposals),
                "camera_motion_max": max((sample.camera_score for sample in samples), default=0.0),
                "detection_trace_cache_hit": detection_trace_cache_hit,
                "coverage_row_type": (
                    proposals[0].get("fallback_type")
                    or "detected_hand_presence"
                    if mode == "presence_recall" and proposals
                    else None
                ),
                "_detection_trace": samples,
            }
        )
        return record, proposals, metrics, None
    except Exception as exc:
        return record, [], {"status": "error", "candidate_count": 0}, repr(exc)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _scan_signature(
    record: SourceRecord,
    args: argparse.Namespace,
    model_sha: str,
    script_sha: str,
    *,
    candidate_start: int,
) -> str:
    source_stat = record.source.stat()
    frame_width, frame_height = _effective_scan_size(args)
    config = {
        "source_rel": record.source_rel,
        "source_sha256": record.source_sha256,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "model_sha256": model_sha,
        "scanner_sha256": script_sha,
        "sample_fps": args.sample_fps,
        "scan_frame_width": frame_width,
        "scan_frame_height": frame_height,
        "threshold": args.threshold,
        "bridge_gap_sec": args.bridge_gap_sec,
        "min_active_sec": args.min_active_sec,
        "pre_roll_sec": args.pre_roll_sec,
        "post_roll_sec": args.post_roll_sec,
        "mode": args.mode,
        "high_recall_max_run_sec": args.high_recall_max_run_sec,
        "high_recall_bridge_gap_sec": args.high_recall_bridge_gap_sec,
        "high_recall_min_active_sec": args.high_recall_min_active_sec,
        "high_recall_quiet_gap_sec": args.high_recall_quiet_gap_sec,
        "presence_bridge_gap_sec": args.presence_bridge_gap_sec,
        "candidate_start": int(candidate_start),
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _detection_signature(
    record: SourceRecord,
    args: argparse.Namespace,
    model_sha: str,
    script_sha: str,
) -> str:
    """Hash inputs that affect detector observations, excluding regrouping."""

    source_stat = record.source.stat()
    frame_width, frame_height = _effective_scan_size(args)
    config = {
        "source_rel": record.source_rel,
        "source_sha256": record.source_sha256,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "model_sha256": model_sha,
        "scanner_sha256": script_sha,
        "sample_fps": args.sample_fps,
        "scan_frame_width": frame_width,
        "scan_frame_height": frame_height,
        "num_hands": 2,
        "min_hand_detection_confidence": 0.35,
        "min_hand_presence_confidence": 0.35,
        "min_tracking_confidence": 0.35,
        "mediapipe_runtime_version": _mediapipe_runtime_version(),
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _mediapipe_runtime_version() -> str:
    """Return the imported MediaPipe runtime version for cache invalidation."""

    try:
        import mediapipe as mp

        return str(getattr(mp, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _effective_scan_size(args: argparse.Namespace) -> tuple[int, int]:
    width = getattr(args, "scan_width", None)
    height = getattr(args, "scan_height", None)
    if width is None and height is None:
        if args.mode == "presence_recall":
            return PRESENCE_FRAME_W, PRESENCE_FRAME_H
        return FRAME_W, FRAME_H
    if width is None or height is None:
        raise ValueError("--scan-width and --scan-height must be set together")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("--scan-width and --scan-height must be positive")
    return width, height


def _checkpoint_path(work_root: Path, record: SourceRecord) -> Path:
    source_key = hashlib.sha256(record.source_rel.encode("utf-8")).hexdigest()
    return work_root / "checkpoints" / f"{source_key}.json"


def _serialize_activity_samples(
    samples: Sequence[ActivitySample],
) -> list[dict[str, Any]]:
    return [
        {
            "time_sec": float(sample.time_sec),
            "score": float(sample.score),
            "hand_count": int(sample.hand_count),
            "camera_score": float(sample.camera_score),
            "hand_centers": [
                [float(x), float(y)] for x, y in sample.hand_centers
            ],
        }
        for sample in samples
    ]


def _deserialize_activity_samples(rows: Any) -> list[ActivitySample]:
    if not isinstance(rows, list):
        raise ValueError("detection_trace must be a list")
    samples: list[ActivitySample] = []
    previous_time: float | None = None
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("detection_trace rows must be objects")
        centers_value = row.get("hand_centers", [])
        if not isinstance(centers_value, list):
            raise ValueError("hand_centers must be a list")
        centers_list: list[tuple[float, float]] = []
        for center in centers_value:
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError("hand center must contain x and y")
            x, y = float(center[0]), float(center[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("hand center coordinates must be finite")
            centers_list.append((x, y))
        centers = tuple(centers_list)
        hand_count = int(row.get("hand_count", len(centers)))
        if hand_count < 0 or hand_count != len(centers):
            raise ValueError("hand_count must match hand_centers")
        sample = ActivitySample(
            time_sec=float(row["time_sec"]),
            score=float(row.get("score", 0.0)),
            hand_count=hand_count,
            camera_score=float(row.get("camera_score", 0.0)),
            hand_centers=centers,
        )
        if not all(
            math.isfinite(value)
            for value in (sample.time_sec, sample.score, sample.camera_score)
        ):
            raise ValueError("detection_trace contains non-finite values")
        if previous_time is not None and sample.time_sec <= previous_time:
            raise ValueError("detection_trace timestamps must be strictly increasing")
        samples.append(sample)
        previous_time = sample.time_sec
    return samples


def _load_checkpoint(
    path: Path, record: SourceRecord, signature: str
) -> tuple[SourceRecord, list[dict[str, Any]], dict[str, Any], None] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        proposals = data["proposals"]
        metrics = data["metrics"]
        if (
            data.get("schema_version") != 2
            or data.get("signature") != signature
            or data.get("source_rel") != record.source_rel
            or data.get("source_sha256") != record.source_sha256
            or not isinstance(proposals, list)
            or not isinstance(metrics, dict)
            or metrics.get("status") != "ok"
            or metrics.get("candidate_count") != len(proposals)
            or any(
                not isinstance(row, dict)
                or row.get("source_rel") != record.source_rel
                or row.get("source_sha256") != record.source_sha256
                for row in proposals
            )
        ):
            return None
        return record, proposals, metrics, None
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _load_detection_trace(
    path: Path,
    record: SourceRecord,
    detection_signature: str,
) -> tuple[list[ActivitySample], dict[str, Any]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        samples = _deserialize_activity_samples(data["detection_trace"])
        if (
            data.get("schema_version") != 2
            or data.get("detection_signature") != detection_signature
            or data.get("source_rel") != record.source_rel
            or data.get("source_sha256") != record.source_sha256
            or not isinstance(metrics, dict)
            or int(metrics.get("sample_count", -1)) != len(samples)
        ):
            return None
        return samples, metrics
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _save_checkpoint(
    path: Path,
    signature: str,
    detection_signature: str,
    result: tuple[SourceRecord, list[dict[str, Any]], dict[str, Any], str | None],
) -> None:
    record, proposals, metrics, error = result
    if error or metrics.get("status") != "ok":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    detection_trace = metrics.get("_detection_trace", [])
    stored_metrics = {
        key: value for key, value in metrics.items() if not key.startswith("_")
    }
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "signature": signature,
                "detection_signature": detection_signature,
                "source_rel": record.source_rel,
                "source_sha256": record.source_sha256,
                "proposals": proposals,
                "metrics": stored_metrics,
                "detection_trace": _serialize_activity_samples(detection_trace),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.inventory:
        records = load_inventory(Path(args.inventory), source_root)
    else:
        records = discover_inventory(source_root)
    if args.limit > 0:
        records = records[: args.limit]
    work_root = Path(args.work_root)
    scan_frame_width, scan_frame_height = _effective_scan_size(args)
    model_sha = sha256_file(model_path)
    script_sha = sha256_file(Path(__file__))
    report_dir = work_root / "reports"
    default_manifest_name = {
        "standard": "causal_proposals.jsonl",
        "high_recall": "causal_proposals_high_recall.jsonl",
        "presence_recall": "hand_presence_proposals.jsonl",
    }[args.mode]
    manifest_path = (
        Path(args.output)
        if args.output
        else work_root / "manifests" / default_manifest_name
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[SourceRecord, list[dict[str, Any]], dict[str, Any], str | None]] = []
    indexed = list(enumerate(records))
    workers = max(1, min(int(args.workers), len(indexed) or 1))
    cache_hit_count = 0
    detection_trace_cache_hit_count = 0

    def scan_or_resume(index: int, record: SourceRecord):
        candidate_start = index * 100000 + 1
        signature = _scan_signature(
            record,
            args,
            model_sha,
            script_sha,
            candidate_start=candidate_start,
        )
        detection_signature = _detection_signature(
            record, args, model_sha, script_sha
        )
        checkpoint = _checkpoint_path(work_root, record)
        cached = _load_checkpoint(checkpoint, record, signature)
        if cached is not None:
            return cached, True, False
        precomputed_scan = _load_detection_trace(
            checkpoint, record, detection_signature
        )
        result = _scan_record(
            record,
            model_path,
            args.sample_fps,
            scan_frame_width,
            scan_frame_height,
            args.threshold,
            args.bridge_gap_sec,
            args.min_active_sec,
            args.pre_roll_sec,
            args.post_roll_sec,
            candidate_start,
            args.mode,
            args.high_recall_max_run_sec,
            args.high_recall_bridge_gap_sec,
            args.high_recall_min_active_sec,
            args.high_recall_quiet_gap_sec,
            args.presence_bridge_gap_sec,
            precomputed_scan,
        )
        _save_checkpoint(checkpoint, signature, detection_signature, result)
        return result, False, precomputed_scan is not None

    if workers == 1:
        for index, record in indexed:
            result, cached, trace_cached = scan_or_resume(index, record)
            results.append(result)
            cache_hit_count += int(cached)
            detection_trace_cache_hit_count += int(trace_cached)
            print(f"[{index + 1}/{len(records)}] {record.source_name}" + (" (cached)" if cached else ""), flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(scan_or_resume, index, record): (index, record)
                for index, record in indexed
            }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result, cached, trace_cached = future.result()
                results.append(result)
                cache_hit_count += int(cached)
                detection_trace_cache_hit_count += int(trace_cached)
                print(f"[{completed}/{len(records)}] {result[0].source_name}" + (" (cached)" if cached else ""), flush=True)
    results.sort(key=lambda item: item[0].source_rel.casefold())
    proposals = [row for _, rows, _, _ in results for row in rows]
    proposals.sort(key=lambda row: (row["source_rel"].casefold(), row["source_start_sec"], row["clip_id"]))
    # Re-index after sorting so IDs are stable across worker counts and resumes.
    for index, row in enumerate(proposals, 1):
        row["proposal_sequence"] = index
    write_jsonl(manifest_path, proposals)
    source_reports = []
    errors = []
    for record, _, metrics, error in results:
        public_metrics = {
            key: value for key, value in metrics.items() if not key.startswith("_")
        }
        row = {
            "source_rel": record.source_rel,
            "source_name": record.source_name,
            **public_metrics,
        }
        if error:
            row["error"] = error
            errors.append({"source_rel": record.source_rel, "error": error})
        source_reports.append(row)
    covered_sources = {str(row.get("source_rel")) for row in proposals}
    uncovered_sources = [
        record.source_rel
        for record in records
        if record.source_rel not in covered_sources
    ]
    coverage_gate_passed = (
        not uncovered_sources if args.mode == "presence_recall" else True
    )
    summary = {
        "proposal_method": {
            "standard": "auto_hand_anchor_v1",
            "high_recall": "auto_hand_anchor_recall_v2",
            "presence_recall": "auto_hand_presence_recall_v1",
        }[args.mode],
        "mode": args.mode,
        "source_root": str(source_root),
        "source_count": len(records),
        "source_error_count": len(errors),
        "proposal_count": len(proposals),
        "coverage_gate_passed": coverage_gate_passed,
        "uncovered_source_count": len(uncovered_sources),
        "uncovered_sources": uncovered_sources,
        "zero_hand_fallback_source_count": len(
            {
                row["source_rel"]
                for row in proposals
                if row.get("zero_hand_detection_fallback") is True
            }
        ),
        "cache_hit_count": cache_hit_count,
        "detection_trace_cache_hit_count": detection_trace_cache_hit_count,
        "review_status": "pending_user_review",
        "semantic_action_verified": False,
        "semantic_feedback_verified": False,
        "feedback_status": "unverified",
        "sample_fps": args.sample_fps,
        "scan_frame_width": scan_frame_width,
        "scan_frame_height": scan_frame_height,
        "bridge_gap_sec": args.bridge_gap_sec,
        "min_active_sec": args.min_active_sec,
        "pre_roll_sec": args.pre_roll_sec,
        "post_roll_sec": args.post_roll_sec,
        "high_recall_max_run_sec": args.high_recall_max_run_sec,
        "high_recall_bridge_gap_sec": args.high_recall_bridge_gap_sec,
        "high_recall_min_active_sec": args.high_recall_min_active_sec,
        "high_recall_quiet_gap_sec": args.high_recall_quiet_gap_sec,
        "presence_bridge_gap_sec": args.presence_bridge_gap_sec,
        "workers": workers,
        "errors": errors,
        "sources": source_reports,
        "output": str(manifest_path),
    }
    (report_dir / "causal_proposal_scan.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--inventory")
    parser.add_argument("--output")
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--scan-width", type=int)
    parser.add_argument("--scan-height", type=int)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--bridge-gap-sec", type=float, default=DEFAULT_BRIDGE_GAP_SEC)
    parser.add_argument("--min-active-sec", type=float, default=DEFAULT_MIN_ACTIVE_SEC)
    parser.add_argument("--pre-roll-sec", type=float, default=DEFAULT_PRE_ROLL_SEC)
    parser.add_argument("--post-roll-sec", type=float, default=DEFAULT_POST_ROLL_SEC)
    parser.add_argument(
        "--mode",
        choices=("standard", "high_recall", "presence_recall"),
        default="standard",
        help=(
            "proposal pass; high_recall uses motion, while presence_recall "
            "only maps hand-visible coverage for manual action review"
        ),
    )
    parser.add_argument(
        "--high-recall-max-run-sec",
        type=float,
        default=HIGH_RECALL_MAX_RUN_SEC,
        help="maximum continuous high-recall continuation length",
    )
    parser.add_argument(
        "--high-recall-bridge-gap-sec",
        type=float,
        default=HIGH_RECALL_BRIDGE_GAP_SEC,
        help="hand-visible gap bridged before high-recall valley splitting",
    )
    parser.add_argument(
        "--high-recall-min-active-sec",
        type=float,
        default=HIGH_RECALL_MIN_ACTIVE_SEC,
        help="minimum high-recall activity run before proposal",
    )
    parser.add_argument(
        "--high-recall-quiet-gap-sec",
        type=float,
        default=HIGH_RECALL_QUIET_GAP_SEC,
        help="minimum low-motion valley used as a high-recall boundary",
    )
    parser.add_argument(
        "--presence-bridge-gap-sec",
        type=float,
        default=PRESENCE_RECALL_BRIDGE_GAP_SEC,
        help="maximum detected-hand anchor gap bridged in presence coverage",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.high_recall_max_run_sec <= 0:
        parser.error("--high-recall-max-run-sec must be positive")
    if args.high_recall_bridge_gap_sec < 0:
        parser.error("--high-recall-bridge-gap-sec must be non-negative")
    if args.high_recall_min_active_sec < 0:
        parser.error("--high-recall-min-active-sec must be non-negative")
    if args.high_recall_quiet_gap_sec < 0:
        parser.error("--high-recall-quiet-gap-sec must be non-negative")
    if args.presence_bridge_gap_sec < 0:
        parser.error("--presence-bridge-gap-sec must be non-negative")
    try:
        summary = run_scan(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "source_count": summary["source_count"],
                "source_error_count": summary["source_error_count"],
                "proposal_count": summary["proposal_count"],
                "output": summary["output"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if summary["source_error_count"] != 0:
        return 2
    if not summary.get("coverage_gate_passed", True):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
