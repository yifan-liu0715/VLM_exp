# Formal V2V Four-Panel Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a deterministic source/target/Base/LoRA four-panel MP4 for `mymotion__002` with the completed formal V2V adapter.

**Architecture:** A focused inference module reuses the exact custom transformer, native T5/VAE loaders, paired preprocessing, and flow scheduler used during training. It runs Base first, loads the final adapter into the same model, reruns from cloned initial noise, then exports individual and labeled composite videos.

**Tech Stack:** Python 3.10, PyTorch, Diffusers/PEFT, safetensors, Decord, Pillow, imageio/ffmpeg.

## Global Constraints

- Project root: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio`.
- Conda environment: `v2v_train`.
- Output root: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/inference_comparison_0716`.
- GPU0 is forbidden; use an idle nonzero GPU.
- Sample is `mymotion__002`; use frames 0-32, 576 x 320, 30 steps, seed 42.
- Do not compute quality metrics.

---

### Task 1: Deterministic preprocessing and panel layout

**Files:**
- Create: `tests/test_infer_v2v_comparison.py`
- Create: `scripts/infer_v2v_comparison.py`

**Interfaces:**
- Produces: `load_manifest_row`, `load_aligned_sample`, `tensor_to_uint8_frames`, and `make_four_panel`.

- [ ] Write tests asserting the selected row, training-bucket dimensions, adapter key normalization, and 2x2 panel geometry.
- [ ] Run `pytest -q tests/test_infer_v2v_comparison.py` and confirm failure because the inference module does not exist.
- [ ] Implement the smallest pure helpers required by the tests.
- [ ] Run the same pytest command and confirm all tests pass.

### Task 2: Base and LoRA inference

**Files:**
- Modify: `scripts/infer_v2v_comparison.py`

**Interfaces:**
- Consumes: formal training config, custom transformer modules, final adapter, and the helpers from Task 1.
- Produces: `run_denoising`, `load_formal_adapter`, and the command-line entry point.

- [ ] Add tests for identical cloned noise and exact adapter tensor classification/count validation.
- [ ] Run pytest and confirm the new assertions fail before implementation.
- [ ] Implement model loading, condition encoding, 30-step Base inference, adapter loading, and 30-step LoRA inference.
- [ ] Run pytest and `python -m py_compile scripts/infer_v2v_comparison.py`; both must exit 0.

### Task 3: GPU execution and artifact verification

**Files:**
- Create: `logs/infer_mymotion__002.log`
- Create: `videos/mymotion__002_{source,target,base,lora,comparison}.mp4`
- Create: `reports/mymotion__002_run_manifest.json`

**Interfaces:**
- Consumes: Task 2 CLI.
- Produces: five MP4 artifacts, a log, and a reproducibility manifest.

- [ ] Query `nvidia-smi`, select an idle nonzero GPU, and record it in the log.
- [ ] Run the script with `CUDA_VISIBLE_DEVICES=<nonzero GPU>` and preserve stdout/stderr.
- [ ] Use `ffprobe` to verify all individual videos are 576 x 320 with 33 frames and the comparison is 1152 x 640 with 33 frames.
- [ ] Verify the run manifest records sample paths, prompt, adapter SHA256, seed, steps, dimensions, and output files.
