"""Small, testable contracts shared by formal paired-V2V training scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import torch


REQUIRED_COLUMNS = {
    "sample_id",
    "bucket",
    "width",
    "height",
    "num_frames",
    "input_video",
    "output_video",
    "reference_image",
    "caption",
}


def absolutize_media_paths(row: Mapping[str, str], dataset_root: str | Path) -> dict[str, str]:
    """Return a copied metadata row whose media fields are absolute paths."""
    root = Path(dataset_root).expanduser().resolve(strict=False)
    converted = dict(row)
    for field in ("input_video", "output_video", "reference_image"):
        path = Path(converted[field]).expanduser()
        if not path.is_absolute():
            path = root / path
        converted[field] = str(path.resolve(strict=False))
    return converted


def convert_original_wan_shard(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert original Wan DiT names without consuming the caller's dictionary."""
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    return convert_wan_transformer_to_diffusers(dict(state_dict))


def build_timestep_sequence(
    timestep: torch.Tensor,
    main_tokens: int,
    source_tokens: int,
    reference_tokens: int = 0,
) -> torch.Tensor:
    """Assign diffusion time to target tokens and time zero to condition tokens."""
    if timestep.ndim != 1:
        raise ValueError(f"timestep must have shape [batch], got {tuple(timestep.shape)}")
    if min(main_tokens, source_tokens, reference_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    if main_tokens == 0:
        raise ValueError("main_tokens must be positive")

    batch = timestep.shape[0]
    target = timestep[:, None].repeat(1, main_tokens)
    conditions = torch.zeros(
        batch,
        source_tokens + reference_tokens,
        device=timestep.device,
        dtype=timestep.dtype,
    )
    return torch.cat((target, conditions), dim=1)


def validate_manifest(path: str | Path, minimum_frames: int = 33) -> list[dict[str, str]]:
    """Validate the immutable paired-video contract before any GPU work starts."""
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("manifest contains no samples")

    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise ValueError(f"line {line_number}: sample_id is empty")
        if sample_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)

        source = Path(row["input_video"]).expanduser().resolve(strict=False)
        target = Path(row["output_video"]).expanduser().resolve(strict=False)
        reference = Path(row["reference_image"]).expanduser().resolve(strict=False)
        if source == target:
            raise ValueError(f"line {line_number}: input_video and output_video must differ")
        for field, media_path in (
            ("input_video", source),
            ("output_video", target),
            ("reference_image", reference),
        ):
            if not media_path.is_file():
                raise ValueError(f"line {line_number}: {field} does not exist: {media_path}")

        try:
            frame_count = int(row["num_frames"])
            width = int(row["width"])
            height = int(row["height"])
        except ValueError as error:
            raise ValueError(f"line {line_number}: invalid integer metadata") from error
        if frame_count < minimum_frames:
            raise ValueError(
                f"line {line_number}: num_frames={frame_count} is below {minimum_frames}"
            )
        if width <= 0 or height <= 0:
            raise ValueError(f"line {line_number}: width and height must be positive")
        if not row["caption"].strip():
            raise ValueError(f"line {line_number}: caption is empty")

    return rows
