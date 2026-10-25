"""Build explicit model commands without executing them."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from mode_comparison.contracts import PROMPT, REMOTE_ROOT, ExperimentContract


SCAIL_PYTHON = "/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python"
LORA_PYTHON = "/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python"
SCAIL_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"
SCAIL_SOURCE = f"{REMOTE_ROOT}/code/SCAIL-2-wan/SCAIL-2-{SCAIL_COMMIT}"
SCAIL_CHECKPOINT = f"{REMOTE_ROOT}/weights/SCAIL-2"


def fairness_contract(contract: ExperimentContract) -> dict[str, object]:
    return {
        "sample_id": contract.sample_id,
        "mode": contract.mode,
        "width": contract.width,
        "height": contract.height,
        "num_frames": contract.output_frames,
        "fps": contract.output_fps,
        "solver": contract.solver,
        "steps": contract.steps,
        "guidance_scale": contract.guidance_scale,
        "shift": contract.shift,
        "seed": contract.seed,
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "shared_input_sha256": "pending_preprocessing",
        "target_usage": "evaluation-only; excluded from generation conditioning",
    }


def build_scail_command(contract: ExperimentContract) -> list[str]:
    root = PurePosixPath(contract.experiment_root)
    inputs = root / "scail2/inputs"
    mask_video = "replace_mask.mp4" if contract.mode == "replacement" else "rendered_mask_v2.mp4"
    output = root / f"scail2/outputs/scail2_{contract.mode}.mp4"
    command = [
        SCAIL_PYTHON,
        f"{SCAIL_SOURCE}/generate.py",
        "--model",
        "SCAIL-14B",
        "--ckpt_dir",
        SCAIL_CHECKPOINT,
        "--scail_path",
        contract.scail2_weight,
        "--scail_config_path",
        f"{SCAIL_SOURCE}/configs/config-14b.json",
        "--prompt",
        PROMPT,
        "--image",
        str(inputs / "reference.png"),
        "--mask_image",
        str(inputs / "ref_mask.png"),
        "--pose",
        str(inputs / "rendered_v2.mp4"),
        "--mask_video",
        str(inputs / mask_video),
    ]
    if contract.mode == "replacement":
        command.append("--replace_flag")
    command += [
        "--target_h",
        str(contract.height),
        "--target_w",
        str(contract.width),
        "--sample_solver",
        contract.solver,
        "--sample_steps",
        str(contract.steps),
        "--sample_shift",
        str(contract.shift),
        "--sample_guide_scale",
        str(contract.guidance_scale),
        "--segment_len",
        str(contract.segment_len),
        "--segment_overlap",
        str(contract.segment_overlap),
        "--base_seed",
        str(contract.seed),
        "--offload_model",
        "true",
        "--save_file",
        str(output),
    ]
    return command


def build_lora_command(contract: ExperimentContract) -> list[str]:
    root = PurePosixPath(contract.experiment_root)
    return [
        LORA_PYTHON,
        str(root / "wan_lora/scripts/infer_scheme4_lora_only.py"),
        "--input-manifest",
        str(root / "wan_lora/manifests/input_manifest.json"),
        "--adapter",
        contract.lora_weight,
        "--expected-adapter-sha256",
        contract.lora_sha256,
        "--output-file",
        str(root / "wan_lora/outputs/lora.mp4"),
        "--run-manifest",
        str(root / "wan_lora/manifests/run_manifest.json"),
    ]
