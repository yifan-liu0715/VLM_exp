"""Teacher-aligned paired V2V dataset backed by an explicit split manifest."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from aspect_fit_geometry import (
    nearest_aligned_canvas,
    resize_reference_preserving_geometry,
    resize_video_preserving_geometry,
)


REQUIRED_COLUMNS = {
    "sample_id",
    "width",
    "height",
    "num_frames",
    "input_video",
    "output_video",
    "reference_image",
    "caption",
    "source_split",
}


def aligned_frame_indices(
    paired_length: int,
    sample_frames: int,
    start: int,
) -> list[int]:
    if paired_length < sample_frames:
        raise ValueError(
            f"paired video is shorter than requested: {paired_length} < {sample_frames}"
        )
    maximum_start = paired_length - sample_frames
    if start < 0 or start > maximum_start:
        raise ValueError(
            f"start={start} is outside [0, {maximum_start}] for the aligned clip"
        )
    return list(range(start, start + sample_frames))


def target_size(
    native_h: int,
    native_w: int,
    max_long_side: int,
    align: int = 32,
) -> tuple[int, int]:
    return nearest_aligned_canvas(
        native_h,
        native_w,
        max_long_side,
        align,
    )


def validate_paired_frame_shapes(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    sample_id: str,
) -> None:
    """Require paired videos to have the same decoded T,H,W,C shape."""
    if tuple(source_shape) != tuple(target_shape):
        raise ValueError(
            f"{sample_id}: Source/Target native frame shapes differ: "
            f"{tuple(source_shape)} vs {tuple(target_shape)}"
        )


def load_manifest_rows(
    manifest_path: str | Path,
    required_split: str | None,
    check_media: bool = True,
    minimum_frames: int = 97,
) -> list[dict[str, str]]:
    path = Path(manifest_path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest contains no rows")

    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise ValueError(f"line {line_number}: empty sample_id")
        if sample_id in seen:
            raise ValueError(f"line {line_number}: duplicate sample_id={sample_id}")
        seen.add(sample_id)
        split = row["source_split"].strip()
        if required_split is not None and split != required_split:
            raise ValueError(
                f"line {line_number}: non-{required_split} row {sample_id} has split={split}"
            )
        if int(row["num_frames"]) < minimum_frames:
            raise ValueError(
                f"line {line_number}: {sample_id} has fewer than {minimum_frames} frames"
            )
        if not row["caption"].strip():
            raise ValueError(f"line {line_number}: {sample_id} has an empty caption")
        if "/output_video/" not in row["input_video"].replace("\\", "/"):
            raise ValueError(
                f"line {line_number}: source path is not physical output_video"
            )
        if "/input_video/" not in row["output_video"].replace("\\", "/"):
            raise ValueError(
                f"line {line_number}: target path is not physical input_video"
            )
        if check_media:
            for field in ("input_video", "output_video", "reference_image"):
                media = Path(row[field]).expanduser()
                if not media.is_file():
                    raise ValueError(
                        f"line {line_number}: missing {field} for {sample_id}: {media}"
                    )
    return rows


try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:  # type: ignore[no-redef]
        """Import fallback so pure contract tests do not require local PyTorch."""


class TeacherAlignedPairedDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sample_frames: int = 97,
        max_long_side: int = 1088,
        bucket_align: int = 32,
        required_split: str = "train",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.sample_frames = int(sample_frames)
        self.max_long_side = int(max_long_side)
        self.bucket_align = int(bucket_align)
        self.rows = load_manifest_rows(
            self.manifest_path,
            required_split=required_split,
            check_media=True,
            minimum_frames=self.sample_frames,
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _resize_video(frames: Any, height: int, width: int):
        return resize_video_preserving_geometry(frames, height, width)

    @staticmethod
    def _resize_reference(path: str, height: int, width: int):
        return resize_reference_preserving_geometry(
            path,
            height=height,
            width=width,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        from decord import VideoReader, cpu
        import torch

        row = self.rows[index]
        source_reader = VideoReader(
            row["input_video"],
            ctx=cpu(0),
            num_threads=1,
        )
        target_reader = VideoReader(
            row["output_video"],
            ctx=cpu(0),
            num_threads=1,
        )
        paired_length = min(
            len(source_reader),
            len(target_reader),
            int(row["num_frames"]),
        )
        maximum_start = paired_length - self.sample_frames
        if maximum_start < 0:
            raise ValueError(
                f"{row['sample_id']}: decoded videos are shorter than "
                f"{self.sample_frames} frames"
            )
        start = (
            int(torch.randint(0, maximum_start + 1, (1,)).item())
            if maximum_start
            else 0
        )
        indices = aligned_frame_indices(
            paired_length,
            self.sample_frames,
            start,
        )
        target_frames = target_reader.get_batch(indices).asnumpy()
        height, width = target_size(
            int(target_frames.shape[1]),
            int(target_frames.shape[2]),
            self.max_long_side,
            self.bucket_align,
        )
        source_frames = source_reader.get_batch(indices).asnumpy()
        validate_paired_frame_shapes(
            tuple(source_frames.shape),
            tuple(target_frames.shape),
            row["sample_id"],
        )
        source = self._resize_video(source_frames, height, width)
        target = self._resize_video(target_frames, height, width)
        reference = self._resize_reference(
            row["reference_image"],
            height,
            width,
        )
        if source.shape != target.shape:
            raise RuntimeError(
                f"{row['sample_id']}: source/target tensors differ: "
                f"{tuple(source.shape)} vs {tuple(target.shape)}"
            )
        return {
            "source_video": source,
            "target_video": target,
            "reference_image": reference,
            "prompt": row["caption"],
            "sample_id": row["sample_id"],
            "source_path": row["input_video"],
            "target_path": row["output_video"],
            "reference_path": row["reference_image"],
            "source_split": row["source_split"],
            "bucket_size": f"{width}x{height}",
        }
