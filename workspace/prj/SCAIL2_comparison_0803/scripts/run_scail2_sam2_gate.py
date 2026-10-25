#!/usr/bin/env python3
"""Fail-closed sequential Base/DPO controller for approved SAM2 fallback masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


VALUELESS_OPTIONS = {"--replace_flag"}
ERROR_PATTERNS = (
    "traceback",
    "cuda out of memory",
    "non-finite",
    "nan detected",
    "no space left on device",
)


def select_runs(plan: dict, sample_id: str) -> list[dict]:
    runs = [item for item in plan.get("runs", []) if item.get("sample_id") == sample_id]
    by_variant = {item.get("variant"): item for item in runs}
    if set(by_variant) != {"base", "dpo"} or not all(
        bool(by_variant[name].get("ready")) for name in ("base", "dpo")
    ):
        raise ValueError(f"sample pair not ready: {sample_id}")
    return [by_variant["base"], by_variant["dpo"]]


def option_map(argv: list[str]) -> tuple[str, dict[str, object]]:
    if not argv:
        raise ValueError("empty argv")
    program = argv[0]
    options: dict[str, object] = {}
    index = 1
    while index < len(argv):
        key = argv[index]
        if not key.startswith("--"):
            raise ValueError(f"unexpected positional argument: {key}")
        if key in options:
            raise ValueError(f"duplicate option: {key}")
        if key in VALUELESS_OPTIONS:
            options[key] = True
            index += 1
        else:
            if index + 1 >= len(argv):
                raise ValueError(f"missing value for {key}")
            options[key] = argv[index + 1]
            index += 2
    return program, options


def assert_fair_pair(base: dict, dpo: dict) -> None:
    base_program, base_options = option_map(base["argv"])
    dpo_program, dpo_options = option_map(dpo["argv"])
    if base_program != dpo_program:
        raise ValueError("fairness mismatch: generator program")
    if "--lora_path" in base_options or "--lora_alpha" in base_options:
        raise ValueError("fairness mismatch: Base unexpectedly contains LoRA options")
    if dpo_options.pop("--lora_alpha", None) != "1.0":
        raise ValueError("fairness mismatch: DPO lora alpha must be 1.0")
    if not dpo_options.pop("--lora_path", None):
        raise ValueError("fairness mismatch: DPO lora path missing")
    base_save = base_options.pop("--save_file", None)
    dpo_save = dpo_options.pop("--save_file", None)
    if not base_save or not dpo_save or base_save == dpo_save:
        raise ValueError("fairness mismatch: Base/DPO outputs must be distinct")
    if base_options != dpo_options:
        changed = sorted(
            key
            for key in set(base_options) | set(dpo_options)
            if base_options.get(key) != dpo_options.get(key)
        )
        raise ValueError(f"fairness mismatch: changed common options {changed}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def _gpu_snapshot(physical_gpu: int) -> dict:
    command = [
        "nvidia-smi",
        f"--id={physical_gpu}",
        "--query-gpu=index,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    row = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    fields = [part.strip() for part in row.split(",")]
    if len(fields) != 4 or int(fields[0]) != physical_gpu:
        raise RuntimeError(f"unexpected physical GPU snapshot: {row!r}")
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "memory_free_mib": int(fields[2]),
        "utilization_gpu": int(fields[3]),
    }


def _scan_log(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    hits = [pattern for pattern in ERROR_PATTERNS if pattern in text]
    if hits:
        raise RuntimeError(f"error patterns in {path}: {hits}")


def _output_path(run: dict) -> Path:
    _, options = option_map(run["argv"])
    value = options.get("--save_file")
    if not value:
        raise ValueError("run is missing --save_file")
    return Path(str(value))


def execute_pair(root: Path, plan: dict, sample_id: str, physical_gpu: int) -> dict:
    if physical_gpu == 0:
        raise ValueError("physical GPU0 is forbidden")
    if plan.get("mask_preprocessor") != "official SAM2 fallback; not official SAM3":
        raise ValueError("plan is not labeled as SAM2 fallback masks")
    runs = select_runs(plan, sample_id)
    assert_fair_pair(runs[0], runs[1])
    gpu = _gpu_snapshot(physical_gpu)
    wan_root = Path(runs[0]["argv"][0]).resolve().parent
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    results = []
    for run in runs:
        variant = run["variant"]
        output = _output_path(run)
        output.parent.mkdir(parents=True, exist_ok=True)
        log = logs_dir / f"scail2_sam2_{sample_id}_{variant}.log"
        command = [run["python"], *run["argv"]]
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=wan_root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{variant} failed with exit {completed.returncode}; see {log}")
        _scan_log(log)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"{variant} did not create a nonempty output: {output}")
        results.append(
            {
                "variant": variant,
                "command": command,
                "log": str(log),
                "output": str(output),
                "size": output.stat().st_size,
                "sha256": _sha256(output),
                "ffprobe": _ffprobe(output),
            }
        )
    pair_dir = _output_path(runs[0]).parent
    manifest = {
        "sample_id": sample_id,
        "status": "completed",
        "mask_preprocessor": "official SAM2 fallback; not official SAM3",
        "target_usage": "evaluation-only; excluded from generation conditioning",
        "physical_gpu": gpu,
        "cuda_visible_devices": str(physical_gpu),
        "fairness_check": "passed",
        "runs": results,
    }
    manifest_path = pair_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = execute_pair(args.root.resolve(), plan, args.sample_id, args.physical_gpu)
    print(json.dumps({"sample_id": args.sample_id, "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
