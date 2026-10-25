"""Filesystem layout and pointer-only weight records."""

from __future__ import annotations

from pathlib import Path

from mode_comparison.contracts import ExperimentContract


def directory_relpaths() -> tuple[str, ...]:
    branches = ("scail2", "wan_lora")
    model_artifacts = ("scripts", "weights", "inputs", "outputs", "logs", "manifests")
    result = [f"{branch}/{artifact}" for branch in branches for artifact in model_artifacts]
    result += [
        "comparison/scripts",
        "comparison/inputs",
        "comparison/outputs",
        "comparison/logs",
        "comparison/manifests",
        "backups/v1",
    ]
    return tuple(result)


def initialize_layout(root: Path) -> tuple[Path, ...]:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"experiment root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    created = []
    for relative in directory_relpaths():
        path = root / relative
        path.mkdir(parents=True, exist_ok=False)
        created.append(path)
    return tuple(created)


def weight_pointer_records(contract: ExperimentContract) -> dict[str, object]:
    return {
        "policy": "pointer_only_no_weight_copy",
        "scail2": {
            "path": contract.scail2_weight,
            "sha256": contract.scail2_sha256,
            "role": "official converted SCAIL-2 inference weight",
        },
        "wan_lora": {
            "path": contract.lora_weight,
            "sha256": contract.lora_sha256,
            "role": "latest Scheme4 trained LoRA adapter",
        },
    }
