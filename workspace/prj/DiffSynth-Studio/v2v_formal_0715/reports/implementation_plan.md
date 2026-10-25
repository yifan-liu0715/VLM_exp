# Formal Paired V2V LoRA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a reproducible Wan2.2-TI2V-5B LoRA on all 3,916 `input_video -> output_video` pairs, using at most three GPUs and never GPU0.

**Architecture:** Adapt the teacher's EffectV2V full-video-conditioning transformer while keeping the existing original Wan checkpoint files. Convert only DiT key names into sharded Diffusers format; load the Wan2.2 VAE through Diffusers' official single-file converter and load the original UMT5 encoder through DiffSynth. Freeze base components and optimize LoRA plus the source-video patch embedder.

**Tech Stack:** PyTorch 2.6, Diffusers 0.38, PyTorch Lightning 2.6, PEFT, Decord/PyAV, OmegaConf, safetensors.

## Global Constraints

- Project root: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio`.
- Formal run root: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715`.
- Dataset uses every row from the frozen rich train/validation/test splits, with `input_video` as condition and `output_video` as target.
- GPU0 is forbidden; use no more than three GPUs selected immediately before launch.
- No inference or validation-video generation is part of this run.
- Only final training scripts remain in the formal scripts directory.

---

### Task 1: Component and data contracts

**Files:**
- Create: `scripts/v2v_components.py`
- Test: `tests/test_v2v_components.py`

- [ ] Run the tests and verify they fail because `v2v_components` is absent.
- [ ] Implement immutable Wan key conversion, timestep construction, and strict paired-manifest validation.
- [ ] Run the focused tests and verify all pass.

### Task 2: Frozen full-data manifest and paired video Dataset

**Files:**
- Create: `scripts/paired_v2v_dataset.py`
- Create: `metadata/metadata_full_paired_0715.csv`
- Create: `metadata/data_manifest_full_0715.json`
- Test: `tests/test_paired_v2v_dataset.py`

- [ ] Test deterministic aligned frame sampling and matching source/target transforms.
- [ ] Implement bucket-aware decoding for 33 aligned frames and resize/crop to a memory-safe training resolution.
- [ ] Freeze all 3,916 audited rows and record SHA256, counts, and bucket distribution.

### Task 3: Original checkpoint loading

**Files:**
- Create: `scripts/prepare_transformer.py`
- Create: `configs/transformer_config.json`
- Create: `configs/vae_config.json`

- [ ] Convert the three original DiT shards with Diffusers' official Wan converter.
- [ ] Load the converted transformer with strict key coverage checks.
- [ ] Load the original Wan2.2 VAE and UMT5 `.pth` without downloading model weights.

### Task 4: Formal Lightning trainer

**Files:**
- Create: `scripts/train_v2v_lora.py`
- Create: `configs/train_v2v_lora_0715.yaml`
- Create: `scripts/launch_train.sh`

- [ ] Implement source-video conditioning, target-only flow-matching loss, LoRA injection, and source patch-embedder training.
- [ ] Save LoRA weights, source patch embedder, optimizer state, resolved config, and step/epoch logs.
- [ ] Disable inference and validation-video export.

### Task 5: Preflight and launch

**Files:**
- Create: `scripts/preflight.py`
- Create: `reports/preflight_report.json`
- Create: `reports/run_manifest.json`

- [ ] Run CPU manifest/config checks.
- [ ] Run one GPU single-batch forward/backward on a free nonzero GPU.
- [ ] Select up to three free nonzero GPUs, launch under `nohup`, and record PID, GPU IDs, command, code hashes, and checkpoint destination.
- [ ] Confirm the process remains alive, loss is finite, GPU0 has no process, and checkpoints are being written.
