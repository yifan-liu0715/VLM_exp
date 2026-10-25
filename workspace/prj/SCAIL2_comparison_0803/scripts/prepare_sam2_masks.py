#!/usr/bin/env python3
"""Prepare target-leakage-free SAM2 fallback masks for one fixed SCAIL-2 sample."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam2_fallback_0808.detector import Detection, detect_people, select_single_or_manual
from sam2_fallback_0808.inputs import assert_conditioning_paths, load_sample
from sam2_fallback_0808.segment import (
    segment_reference,
    select_source_track,
    validate_propagated_masks,
)


SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate official-SAM2 fallback masks (not official SAM3 preprocessing)."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--manual-box-file", type=Path)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _first_video_frame(path: Path) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {path}")
    try:
        ok, bgr = capture.read()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not ok or bgr is None:
        raise ValueError(f"cannot read source frame 0: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fps, frame_count


def _load_manual_boxes(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"reference_box", "source_box"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown manual box keys: {sorted(unknown)}")
    return payload


def _serialize_detection(value: Detection) -> dict:
    record = asdict(value)
    record["box_xyxy"] = list(map(float, value.box_xyxy))
    return record


def _autocast_context(device: str):
    if not device.startswith("cuda"):
        return contextlib.nullcontext()
    import torch

    return torch.autocast("cuda", dtype=torch.bfloat16)


def _build_image_predictor(checkpoint: Path, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(SAM2_CONFIG, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def _build_video_predictor(checkpoint: Path, device: str):
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(SAM2_CONFIG, str(checkpoint), device=device)


def _choose_source_box(
    predictor,
    reference_mask: np.ndarray,
    source_frame: np.ndarray,
    detections: list[Detection],
    manual_box,
) -> tuple[Detection, list[float] | None]:
    canvas = (source_frame.shape[1], source_frame.shape[0])
    if manual_box is not None or len(detections) <= 1:
        return select_single_or_manual(detections, canvas, manual_box), None
    resized_reference = cv2.resize(
        reference_mask.astype(np.uint8), canvas, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    candidate_masks = [
        segment_reference(predictor, source_frame, detection.box_xyxy)
        for detection in detections
    ]
    index, scores = select_source_track(resized_reference, candidate_masks)
    return detections[index], scores


def _propagate_video_masks(predictor, video_path: Path, box, expected_frames: int):
    import torch

    state = predictor.init_state(video_path=str(video_path), offload_video_to_cpu=True)
    try:
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=np.asarray(box, dtype=np.float32),
        )
        by_frame: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for frame_index, object_ids, logits in predictor.propagate_in_video(state):
                if frame_index >= expected_frames:
                    break
                ids = [int(value) for value in object_ids]
                if 1 not in ids:
                    raise ValueError(f"tracked object missing at frame {frame_index}")
                mask_index = ids.index(1)
                mask = (logits[mask_index] > 0.0).detach().cpu().numpy()
                by_frame[int(frame_index)] = np.asarray(mask).squeeze().astype(bool)
    finally:
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)
    missing = [index for index in range(expected_frames) if index not in by_frame]
    if missing:
        raise ValueError(f"missing propagated frames: {missing[:8]}")
    return [by_frame[index] for index in range(expected_frames)]


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    sample = load_sample(root, args.sample_id)
    assert_conditioning_paths(sample, [sample.source_video, sample.reference_image])
    manual = _load_manual_boxes(args.manual_box_file)

    output = root / "preprocessed_sam2" / sample.sample_id
    raw_masks = output / "raw_masks"
    video_masks_dir = raw_masks / "video"
    output.mkdir(parents=True, exist_ok=True)

    reference_rgb = _read_rgb(sample.reference_image)
    source_frame, observed_fps, observed_frames = _first_video_frame(sample.source_video)
    if source_frame.shape[:2] != (sample.height, sample.width):
        raise ValueError(
            f"source dimensions {source_frame.shape[1]}x{source_frame.shape[0]} do not match "
            f"manifest {sample.width}x{sample.height}"
        )
    if observed_frames < sample.num_frames:
        raise ValueError(f"source has only {observed_frames} frames; need {sample.num_frames}")
    if abs(observed_fps - sample.fps) > 0.01:
        raise ValueError(f"source fps {observed_fps} != manifest {sample.fps}")

    yolox = root / "weights/sam2_fallback/yolox_l.onnx"
    checkpoint = root / "weights/sam2_fallback/sam2.1_hiera_large.pt"
    if not yolox.is_file():
        raise FileNotFoundError(yolox)
    reference_detections = detect_people(reference_rgb, yolox)
    source_detections = detect_people(source_frame, yolox)
    detection_record = {
        "sample_id": sample.sample_id,
        "mask_preprocessor": "official SAM2 fallback; not official SAM3",
        "conditioning_paths": [str(sample.source_video), str(sample.reference_image)],
        "reference_candidates": [_serialize_detection(x) for x in reference_detections],
        "source_frame_0_candidates": [_serialize_detection(x) for x in source_detections],
    }
    serialized = json.dumps(detection_record, indent=2, ensure_ascii=False)
    if str(sample.target_video) in serialized:
        raise ValueError("target leakage in detector evidence")
    (output / "detector_candidates.json").write_text(serialized + "\n", encoding="utf-8")
    if args.detect_only:
        return 0
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    with torch.inference_mode(), _autocast_context(args.device):
        image_predictor = _build_image_predictor(checkpoint, args.device)
        reference_actor = select_single_or_manual(
            reference_detections,
            (reference_rgb.shape[1], reference_rgb.shape[0]),
            manual.get("reference_box"),
        )
        reference_mask = segment_reference(
            image_predictor, reference_rgb, reference_actor.box_xyxy
        )
        source_actor, match_scores = _choose_source_box(
            image_predictor,
            reference_mask,
            source_frame,
            source_detections,
            manual.get("source_box"),
        )
        del image_predictor
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        video_predictor = _build_video_predictor(checkpoint, args.device)
        masks = _propagate_video_masks(
            video_predictor,
            sample.source_video,
            source_actor.box_xyxy,
            sample.num_frames,
        )
        del video_predictor
    masks = validate_propagated_masks(
        masks, sample.num_frames, (sample.height, sample.width)
    )

    video_masks_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(reference_mask.astype(np.uint8) * 255).save(raw_masks / "reference.png")
    for index, mask in enumerate(masks):
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            video_masks_dir / f"frame_{index:05d}.png"
        )
    evidence = {
        **detection_record,
        "reference_actor": _serialize_detection(reference_actor),
        "source_actor": _serialize_detection(source_actor),
        "source_match_iou_scores": match_scores,
        "raw_reference_mask": str(raw_masks / "reference.png"),
        "raw_video_mask_dir": str(video_masks_dir),
        "frame_count": len(masks),
        "fps": sample.fps,
        "width": sample.width,
        "height": sample.height,
    }
    evidence_text = json.dumps(evidence, indent=2, ensure_ascii=False)
    if str(sample.target_video) in evidence_text:
        raise ValueError("target leakage in mask evidence")
    (output / "raw_mask_manifest.json").write_text(evidence_text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
