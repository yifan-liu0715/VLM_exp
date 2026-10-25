#!/usr/bin/env python3
"""Audit one completed mode comparison and create labeled visual artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from mode_comparison.audit import (
    compare_fairness_fields,
    normalize_probe,
    parse_physical_gpu,
    scan_log_errors,
)
from mode_comparison.contracts import comparison_contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def probe_video(path: Path, ffprobe: str) -> dict[str, object]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return normalize_probe(json.loads(result.stdout)["streams"][0])


def _create_grid(inputs: list[Path], output: Path, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = ("SOURCE", "TARGET", "SCAIL-2 REPLACEMENT", "LATEST LORA")
    filters = []
    for index, label in enumerate(labels):
        filters.append(
            f"[{index}:v]drawtext=text='{label}':x=12:y=12:fontsize=28:"
            f"fontcolor=white:box=1:boxcolor=black@0.75[v{index}]"
        )
    filters += [
        "[v0][v1]hstack=inputs=2[top]",
        "[v2][v3]hstack=inputs=2[bottom]",
        "[top][bottom]vstack=inputs=2[out]",
    ]
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for path in inputs:
        command += ["-i", str(path)]
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-frames:v",
        "65",
        "-r",
        "16",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)


def _create_contact_sheet(grid: Path, output: Path, ffmpeg: str) -> None:
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(grid),
            "-vf",
            "select='eq(n,0)+eq(n,32)+eq(n,64)',scale=512:928,tile=3x1",
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )


def audit(args: argparse.Namespace) -> dict:
    contract = comparison_contract(args.mode)
    root = args.root.resolve()
    plan_path = root / "comparison/manifests/run_plan.json"
    scail_input_path = root / "scail2/manifests/input_manifest.json"
    lora_input_path = root / "wan_lora/manifests/input_manifest.json"
    lora_run_path = root / "wan_lora/manifests/run_manifest.json"
    scail_log_path = root / "scail2/logs/run.log"
    lora_log_path = root / "wan_lora/logs/run.log"
    source = root / "comparison/inputs/source.mp4"
    target = root / "comparison/inputs/target.mp4"
    scail_output = root / f"scail2/outputs/scail2_{contract.mode}.mp4"
    lora_output = root / "wan_lora/outputs/lora.mp4"
    required = (
        plan_path,
        scail_input_path,
        lora_input_path,
        lora_run_path,
        scail_log_path,
        lora_log_path,
        source,
        target,
        scail_output,
        lora_output,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"comparison artifacts are missing: {missing}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scail_input = json.loads(scail_input_path.read_text(encoding="utf-8"))
    lora_input = json.loads(lora_input_path.read_text(encoding="utf-8"))
    lora_run = json.loads(lora_run_path.read_text(encoding="utf-8"))
    if plan.get("mode") != contract.mode:
        raise ValueError("run plan mode mismatch")
    if contract.mode == "replacement" and "--replace_flag" not in plan["scail2"]["argv"]:
        raise ValueError("replacement run plan is missing --replace_flag")
    if lora_run.get("variant") != "latest_lora" or lora_run.get("status") != "passed":
        raise ValueError("latest LoRA run manifest did not pass")
    if lora_input["target_video"] in lora_input.get("generation_conditioning", []):
        raise ValueError("target leakage in LoRA generation conditioning")

    scail_log = scail_log_path.read_text(encoding="utf-8", errors="replace")
    lora_log = lora_log_path.read_text(encoding="utf-8", errors="replace")
    log_errors = {
        "scail2": scan_log_errors(scail_log),
        "wan_lora": scan_log_errors(lora_log),
    }
    if any(log_errors.values()):
        raise ValueError(f"failure signatures in logs: {log_errors}")
    if contract.mode == "replacement" and "Mode: Replacement" not in scail_log:
        raise ValueError("SCAIL log does not confirm Replacement mode")
    scail_gpu = parse_physical_gpu(scail_log)
    lora_gpu = str(lora_run.get("physical_gpu", ""))
    if not lora_gpu.isdigit() or lora_gpu == "0":
        raise ValueError(f"invalid LoRA physical GPU record: {lora_gpu}")

    expected_probe = {"width": 512, "height": 928, "fps": 16.0, "frames": 65}
    probes = {
        "source": probe_video(source, args.ffprobe),
        "target": probe_video(target, args.ffprobe),
        "scail2": probe_video(scail_output, args.ffprobe),
        "wan_lora": probe_video(lora_output, args.ffprobe),
    }
    bad_probes = {key: value for key, value in probes.items() if value != expected_probe}
    if bad_probes:
        raise ValueError(f"video contract failures: {bad_probes}")

    if sha256_file(root / "scail2/inputs/driving.mp4") != sha256_file(
        root / "wan_lora/inputs/source.mp4"
    ):
        raise ValueError("SCAIL and LoRA source inputs are not byte-identical")
    if sha256_file(root / "scail2/inputs/reference.png") != sha256_file(
        root / "wan_lora/inputs/reference.png"
    ):
        raise ValueError("SCAIL and LoRA reference inputs are not byte-identical")

    scail_fairness = dict(plan["fairness"])
    lora_fairness = {
        "prompt": lora_run["prompt"],
        "width": lora_run["width"],
        "height": lora_run["height"],
        "num_frames": lora_run["num_frames"],
        "fps": lora_run["fps"],
        "solver": "unipc",
        "steps": lora_run["steps"],
        "guidance_scale": lora_run["guidance_scale"],
        "shift": lora_run["flow_shift"],
        "seed": lora_run["seed"],
    }
    differences = compare_fairness_fields(scail_fairness, lora_fairness)
    if differences:
        raise ValueError(f"fairness contract differs: {differences}")

    scail_manifest = {
        "status": "passed",
        "mode": contract.mode,
        "physical_gpu": scail_gpu,
        "input_manifest": str(scail_input_path),
        "output": str(scail_output),
        "output_sha256": sha256_file(scail_output),
        **{key: scail_fairness[key] for key in (
            "prompt", "width", "height", "num_frames", "fps", "solver",
            "steps", "guidance_scale", "shift", "seed"
        )},
    }
    scail_run_path = root / "scail2/manifests/run_manifest.json"
    atomic_json(scail_run_path, scail_manifest)

    grid = root / f"comparison/outputs/{contract.mode}_vs_lora_4grid.mp4"
    contact_sheet = root / f"comparison/outputs/{contract.mode}_frames_0_32_64.png"
    if grid.exists() or contact_sheet.exists():
        raise FileExistsError("refusing to overwrite comparison visual outputs")
    _create_grid([source, target, scail_output, lora_output], grid, args.ffmpeg)
    grid_probe = probe_video(grid, args.ffprobe)
    expected_grid_probe = {"width": 1024, "height": 1856, "fps": 16.0, "frames": 65}
    if grid_probe != expected_grid_probe:
        raise ValueError(f"comparison grid probe {grid_probe}, expected {expected_grid_probe}")
    _create_contact_sheet(grid, contact_sheet, args.ffmpeg)

    final = {
        "status": "passed",
        "mode": contract.mode,
        "fairness_differences": differences,
        "mask_preprocessor": scail_input["mask_preprocessor"],
        "physical_gpus": {"scail2": scail_gpu, "wan_lora": lora_gpu},
        "probes": probes,
        "log_errors": log_errors,
        "outputs": {
            "scail2": str(scail_output),
            "scail2_sha256": sha256_file(scail_output),
            "wan_lora": str(lora_output),
            "wan_lora_sha256": sha256_file(lora_output),
            "comparison": str(grid),
            "comparison_sha256": sha256_file(grid),
            "contact_sheet": str(contact_sheet),
            "contact_sheet_sha256": sha256_file(contact_sheet),
        },
        "large_weights_copied": False,
        "old_experiments_modified": False,
    }
    atomic_json(root / "comparison/manifests/final_audit.json", final)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["replacement", "animation"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def main() -> None:
    result = audit(parse_args())
    print(json.dumps({"status": result["status"], "mode": result["mode"]}))


if __name__ == "__main__":
    main()
