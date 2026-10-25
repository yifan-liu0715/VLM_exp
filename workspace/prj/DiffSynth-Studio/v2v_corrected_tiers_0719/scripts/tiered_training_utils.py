"""Pure contracts for corrected-direction tiered V2V training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


MILESTONES: dict[int, str] = {
    3265: "quick",
    6530: "conservative",
    20000: "recommended",
}


def swap_v2v_direction(row: Mapping[str, str]) -> dict[str, str]:
    """Return a copy with the prior input/output video roles exchanged."""
    source = row["input_video"]
    target = row["output_video"]
    if source == target:
        raise ValueError("input_video and output_video must differ")
    corrected = dict(row)
    corrected["input_video"] = target
    corrected["output_video"] = source
    return corrected


def validate_milestones(milestones: Mapping[int, str], max_steps: int) -> None:
    """Validate an ordered, unique tier schedule ending exactly at max_steps."""
    if not milestones:
        raise ValueError("at least one milestone is required")
    steps = list(milestones)
    names = list(milestones.values())
    if steps != sorted(steps) or any(step <= 0 for step in steps):
        raise ValueError("milestone steps must be positive and increasing")
    if len(names) != len(set(names)):
        raise ValueError("milestone tier names must be unique")
    if steps[-1] != max_steps:
        raise ValueError("final milestone must equal max_steps")


def milestone_adapter_path(
    output_root: str | Path,
    step: int,
    milestones: Mapping[int, str] = MILESTONES,
) -> Path:
    """Return the immutable adapter path for a configured training step."""
    try:
        tier = milestones[step]
    except KeyError as error:
        raise KeyError(f"step {step} is not configured as a milestone") from error
    return (
        Path(output_root)
        / tier
        / "final_adapter"
        / "v2v_lora_and_source_patch.safetensors"
    )
