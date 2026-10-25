#!/usr/bin/env python3
"""Build an isolated 32-aligned low-resolution SCAIL-2 Base/DPO plan."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
from pathlib import Path

_BASE_PLAN_PATH = Path(__file__).with_name("build_scail2_inference_plan.py")
_BASE_PLAN_SPEC = importlib.util.spec_from_file_location("scail2_base_plan", _BASE_PLAN_PATH)
_BASE_PLAN_MODULE = importlib.util.module_from_spec(_BASE_PLAN_SPEC)
assert _BASE_PLAN_SPEC.loader is not None
_BASE_PLAN_SPEC.loader.exec_module(_BASE_PLAN_MODULE)
build_plan = _BASE_PLAN_MODULE.build_plan


LOWRES_CANVASES = {
    (1280, 512): (960, 384),
    (1280, 704): (928, 512),
    (704, 1280): (512, 928),
}


def _option_index(argv: list[str], name: str) -> int:
    try:
        return argv.index(name)
    except ValueError as exc:
        raise ValueError(f"missing required option: {name}") from exc


def _replace_option(argv: list[str], name: str, value: str) -> None:
    index = _option_index(argv, name)
    argv[index + 1] = value


def build_lowres_plan(root: Path, segment_len: int = 65) -> dict:
    plan = build_plan(root, segment_len=segment_len)
    output_root = root / "outputs" / "sam2_lowres_base_vs_dpo"

    for run in plan["runs"]:
        argv = run["argv"]
        width = int(argv[_option_index(argv, "--target_w") + 1])
        height = int(argv[_option_index(argv, "--target_h") + 1])
        try:
            lowres_width, lowres_height = LOWRES_CANVASES[(width, height)]
        except KeyError as exc:
            raise ValueError(f"unsupported source canvas: {width}x{height}") from exc
        _replace_option(argv, "--target_w", str(lowres_width))
        _replace_option(argv, "--target_h", str(lowres_height))
        _replace_option(
            argv,
            "--save_file",
            str(output_root / run["sample_id"] / f"{run['variant']}.mp4"),
        )
        run["command"] = shlex.join([run["python"], *argv])

    plan["resolution_policy"] = "lowres_32_aligned_aspect_preserving"
    plan["lowres_canvases"] = {
        f"{width}x{height}": {"width": low_width, "height": low_height}
        for (width, height), (low_width, low_height) in LOWRES_CANVASES.items()
    }
    plan["output_root"] = str(output_root)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-len", type=int, default=65)
    args = parser.parse_args()
    plan = build_lowres_plan(args.root.resolve(), segment_len=args.segment_len)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "runs": len(plan["runs"])}))


if __name__ == "__main__":
    main()
