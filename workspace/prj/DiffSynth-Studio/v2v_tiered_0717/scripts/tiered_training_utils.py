"""Pure helpers for exact tiered-training milestone exports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def validate_milestones(
    milestones: Mapping[int, str], *, max_steps: int
) -> dict[int, str]:
    normalized = {int(step): str(label) for step, label in milestones.items()}
    if not normalized or any(step <= 0 for step in normalized):
        raise ValueError("milestone steps must be positive")
    if any(not label.strip() for label in normalized.values()):
        raise ValueError("milestone labels must be non-empty")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("milestone labels must be unique")
    if max(normalized) != int(max_steps):
        raise ValueError("last milestone must equal max_steps")
    return dict(sorted(normalized.items()))


def milestone_adapter_path(
    output_root: str | Path,
    milestones: Mapping[int, str],
    step: int,
) -> Path:
    try:
        label = milestones[int(step)]
    except KeyError as error:
        raise ValueError(f"step {step} is not a configured milestone") from error
    return (
        Path(output_root)
        / label
        / "final_adapter"
        / "v2v_lora_and_source_patch.safetensors"
    )
