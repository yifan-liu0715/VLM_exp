"""Reference-only trainable position encoding for Scheme 4."""

from __future__ import annotations

import torch
from torch import nn


class FactorizedReferencePositionEncoding(nn.Module):
    """Additive factorized position tables for patched Reference features."""

    def __init__(
        self,
        dim: int,
        max_frames: int = 4,
        max_height: int = 64,
        max_width: int = 64,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        values = (dim, max_frames, max_height, max_width)
        if min(values) <= 0:
            raise ValueError(f"position dimensions must be positive: {values}")
        if init_std <= 0:
            raise ValueError("init_std must be positive")

        self.temporal = nn.Parameter(torch.empty(max_frames, dim))
        self.height = nn.Parameter(torch.empty(max_height, dim))
        self.width = nn.Parameter(torch.empty(max_width, dim))
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=float(init_std))

    def forward(self, patched_reference: torch.Tensor) -> torch.Tensor:
        if patched_reference.ndim != 5:
            raise ValueError(
                "patched Reference must have shape [B,D,T,H,W], got "
                f"{tuple(patched_reference.shape)}"
            )
        _, dim, frames, height, width = patched_reference.shape
        expected_dim = self.temporal.shape[1]
        if dim != expected_dim:
            raise ValueError(
                f"Reference channel dimension {dim} != table dimension "
                f"{expected_dim}"
            )
        limits = (
            self.temporal.shape[0],
            self.height.shape[0],
            self.width.shape[0],
        )
        actual = (frames, height, width)
        if any(size > limit for size, limit in zip(actual, limits)):
            raise ValueError(
                f"Reference latent patch grid {actual} exceeds position "
                f"table limits {limits}"
            )

        position = (
            self.temporal[:frames, None, None]
            + self.height[None, :height, None]
            + self.width[None, None, :width]
        )
        return (
            position.reshape(1, frames * height * width, dim)
            .to(
                device=patched_reference.device,
                dtype=patched_reference.dtype,
            )
            .contiguous()
        )
