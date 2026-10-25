"""Pure helpers for formal paired-V2V flow-matching training."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def materialize_vae_statistics(model, mean: list[float], std: list[float]) -> None:
    """Restore non-buffer Wan VAE normalization tensors after meta-device loading."""
    model.mean = torch.tensor(mean, dtype=torch.float32)
    model.std = torch.tensor(std, dtype=torch.float32)
    model.scale = [model.mean, 1.0 / model.std]


def filter_trainable_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop all frozen base weights from a Lightning checkpoint."""
    return {
        key: value
        for key, value in state_dict.items()
        if ".lora_" in key or key.startswith("transformer.patch_embedding_extra.")
    }


def flow_match_sample(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create rectified-flow noisy latents and the velocity target."""
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {clean.shape} vs {noise.shape}")
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    sigma = sigma.to(device=clean.device, dtype=clean.dtype)
    noisy = (1.0 - sigma) * clean + sigma * noise
    return noisy, noise - clean
