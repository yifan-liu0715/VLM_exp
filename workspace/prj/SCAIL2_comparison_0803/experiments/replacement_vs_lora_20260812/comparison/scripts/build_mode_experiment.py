#!/usr/bin/env python3
"""Create a new empty mode experiment and pointer-only run plan."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from mode_comparison.commands import (
    build_lora_command,
    build_scail_command,
    fairness_contract,
)
from mode_comparison.contracts import comparison_contract
from mode_comparison.layout import initialize_layout, weight_pointer_records


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_experiment(root: Path, mode: str) -> dict[str, object]:
    contract = comparison_contract(mode)
    root = Path(root)
    if root.name != contract.experiment_name:
        raise ValueError(
            f"experiment root name must be {contract.experiment_name}, got {root.name}"
        )
    initialize_layout(root)
    weights = weight_pointer_records(contract)
    scail_pointer = {
        "policy": weights["policy"],
        "weight": weights["scail2"],
    }
    lora_pointer = {
        "policy": weights["policy"],
        "weight": weights["wan_lora"],
    }
    atomic_json(root / "scail2/weights/weights.json", scail_pointer)
    atomic_json(root / "wan_lora/weights/weights.json", lora_pointer)

    scail_argv = build_scail_command(contract)
    lora_argv = build_lora_command(contract)
    plan = {
        "status": "awaiting_inputs",
        "mode": contract.mode,
        "experiment_root": contract.experiment_root,
        "fairness": fairness_contract(contract),
        "execution_order": ["prepare_inputs", "scail2", "wan_lora", "audit"],
        "scail2": {
            "argv": scail_argv,
            "command": shlex.join(scail_argv),
            "weight_pointer": str(root / "scail2/weights/weights.json"),
        },
        "wan_lora": {
            "argv": lora_argv,
            "command": shlex.join(lora_argv),
            "weight_pointer": str(root / "wan_lora/weights/weights.json"),
        },
        "preservation": {
            "old_experiments": "read-only",
            "large_weight_copy": False,
            "large_video_git_tracking": False,
        },
    }
    atomic_json(root / "comparison/manifests/run_plan.json", plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["replacement", "animation"], required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.root is None:
        args.root = Path(comparison_contract(args.mode).experiment_root)
    return args


def main() -> None:
    args = parse_args()
    result = build_experiment(args.root, args.mode)
    print(json.dumps({"status": result["status"], "mode": result["mode"]}))


if __name__ == "__main__":
    main()
