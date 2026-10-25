"""Inference-only Reference position-table extrapolation for native canvases."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def _interpolate_table_prefix(
    table: nn.Parameter,
    *,
    used_length: int,
    target_length: int,
) -> nn.Parameter:
    if min(int(used_length), int(target_length)) <= 0:
        raise ValueError("position grid dimensions must be positive")
    if int(used_length) > int(table.shape[0]):
        raise ValueError(
            f"trained position length {int(used_length)} exceeds "
            f"table length {int(table.shape[0])}"
        )
    source = (
        table[: int(used_length)]
        .detach()
        .to(torch.float32)
        .transpose(0, 1)
        .unsqueeze(0)
    )
    resized = functional.interpolate(
        source,
        size=int(target_length),
        mode="linear",
        align_corners=True,
    )
    values = (
        resized.squeeze(0)
        .transpose(0, 1)
        .to(device=table.device, dtype=table.dtype)
        .contiguous()
    )
    return nn.Parameter(values, requires_grad=table.requires_grad)


def interpolate_reference_position_for_native_grid(
    module: nn.Module,
    *,
    trained_grid: tuple[int, int],
    native_grid: tuple[int, int],
) -> dict[str, list[int] | str]:
    """Resize learned Scheme 4 H/W table prefixes to a native patch grid."""
    trained_height, trained_width = map(int, trained_grid)
    native_height, native_width = map(int, native_grid)
    if min(
        trained_height,
        trained_width,
        native_height,
        native_width,
    ) <= 0:
        raise ValueError("position grid dimensions must be positive")
    module.height = _interpolate_table_prefix(
        module.height,
        used_length=trained_height,
        target_length=native_height,
    )
    module.width = _interpolate_table_prefix(
        module.width,
        used_length=trained_width,
        target_length=native_width,
    )
    return {
        "trained_grid": [trained_height, trained_width],
        "native_grid": [native_height, native_width],
        "mode": "linear_align_corners",
    }

