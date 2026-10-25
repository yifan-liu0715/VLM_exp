"""Install the Scheme 4 Reference-only position encoder on a Transformer."""

from __future__ import annotations

import torch

from models.wan2.reference_position_encoding import (
    FactorizedReferencePositionEncoding,
)


def install_reference_position_encoding(
    transformer: torch.nn.Module,
    cfg,
    *,
    trainable: bool,
    enabled: bool,
) -> FactorizedReferencePositionEncoding:
    section = cfg.reference_position
    patch_weight = transformer.patch_embedding.weight
    module = FactorizedReferencePositionEncoding(
        dim=int(transformer.patch_embedding.out_channels),
        max_frames=int(section.max_frames),
        max_height=int(section.max_height),
        max_width=int(section.max_width),
        init_std=float(section.init_std),
    ).to(device=patch_weight.device, dtype=patch_weight.dtype)
    module.requires_grad_(bool(trainable))
    transformer.reference_position_encoder = module
    transformer.reference_position_encoding_enabled = bool(enabled)
    return module
