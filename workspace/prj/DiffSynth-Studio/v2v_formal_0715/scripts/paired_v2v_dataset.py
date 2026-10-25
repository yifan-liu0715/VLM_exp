"""Paired full-video Dataset for source-video + reference-image -> target-video training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from v2v_components import validate_manifest


BUCKET_TARGET_SIZES = {
    "1280x512": (256, 640),
    "1280x704": (320, 576),
    "704x1280": (576, 320),
}


def target_size_for_bucket(bucket: str) -> tuple[int, int]:
    try:
        return BUCKET_TARGET_SIZES[bucket]
    except KeyError as error:
        raise ValueError(f"unsupported bucket: {bucket}") from error


def aligned_frame_indices(total_frames: int, sample_frames: int, start: int) -> list[int]:
    if total_frames < sample_frames:
        raise ValueError(
            f"video has {total_frames} frames, fewer than required {sample_frames}"
        )
    maximum_start = total_frames - sample_frames
    if start < 0 or start > maximum_start:
        raise ValueError(f"start={start} is outside [0, {maximum_start}]")
    return list(range(start, start + sample_frames))


def normalize_video(frames: torch.Tensor) -> torch.Tensor:
    return frames.to(torch.float32).div_(127.5).sub_(1.0)


def _resize_video(frames_fhwc: np.ndarray, height: int, width: int) -> torch.Tensor:
    frames = torch.from_numpy(frames_fhwc).permute(0, 3, 1, 2).contiguous()
    frames = normalize_video(frames)
    frames = F.interpolate(
        frames,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return frames.permute(1, 0, 2, 3).contiguous()


def _load_reference(path: str, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    tensor = normalize_video(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


class PairedV2VDataset(Dataset):
    def __init__(self, manifest_path: str | Path, sample_frames: int = 33):
        self.manifest_path = Path(manifest_path)
        self.sample_frames = sample_frames
        self.rows = validate_manifest(self.manifest_path, minimum_frames=sample_frames)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        from decord import VideoReader, cpu

        row = self.rows[index]
        source_reader = VideoReader(row["input_video"], ctx=cpu(0), num_threads=1)
        target_reader = VideoReader(row["output_video"], ctx=cpu(0), num_threads=1)
        paired_length = min(len(source_reader), len(target_reader), int(row["num_frames"]))
        maximum_start = paired_length - self.sample_frames
        if maximum_start < 0:
            raise ValueError(f"{row['sample_id']}: decoded videos are shorter than requested")
        start = int(torch.randint(0, maximum_start + 1, (1,)).item()) if maximum_start else 0
        indices = aligned_frame_indices(paired_length, self.sample_frames, start)

        height, width = target_size_for_bucket(row["bucket"])
        source = _resize_video(source_reader.get_batch(indices).asnumpy(), height, width)
        target = _resize_video(target_reader.get_batch(indices).asnumpy(), height, width)
        reference = _load_reference(row["reference_image"], height, width)
        if source.shape != target.shape:
            raise RuntimeError(
                f"{row['sample_id']}: paired tensors differ: {source.shape} vs {target.shape}"
            )

        return {
            "source_video": source,
            "target_video": target,
            "reference_image": reference,
            "prompt": row["caption"],
            "sample_id": row["sample_id"],
            "bucket": row["bucket"],
        }
