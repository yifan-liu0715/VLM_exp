# Wan V2V Reference Geometry Scheme 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and validate an isolated 2,000-step Wan V2V LoRA that changes only the external Reference geometry preprocessing.

**Architecture:** Copy the completed teacher-aligned experiment's executable code into a new isolated experiment root. Add one shared `reference_geometry.py` module, then route both dataset training and inference through it while retaining the original Source/Target resize implementation unchanged.

**Tech Stack:** Python 3.11, PyTorch, Pillow, NumPy, PyTorch Lightning, DiffSynth/Wan2.2, pytest/unittest, ffprobe.

## Global Constraints

- Remote project root is `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio`.
- Scheme-1 root is `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_reference_aspect_scheme1_0727`.
- Train exactly 2,000 global steps.
- Physical GPU 0 is forbidden.
- Use at most one nonzero physical GPU for this diagnostic.
- Source/Target preprocessing must remain unchanged.
- Training and inference must use the same Reference preprocessing function.
- Scheme 2 must not be created or run.
- Copy verified results to `D:\Code\Wan2.2_v2v_reference_aspect_scheme1_<timestamp>`.

---

### Task 1: Freeze the isolated experiment inputs

**Files:**
- Create: `v2v_reference_aspect_scheme1_0727/configs/train_scheme1_2000.yaml`
- Create: `v2v_reference_aspect_scheme1_0727/metadata/{train,val,test}_manifest.csv`
- Create: `v2v_reference_aspect_scheme1_0727/scripts/*.py`
- Create: `v2v_reference_aspect_scheme1_0727/teacher_reference/**`

**Interfaces:**
- Consumes: completed teacher-aligned experiment scripts and manifests.
- Produces: an isolated, auditable source tree with no checkpoint reuse.

- [ ] Copy only code, configs, manifests, and teacher reference implementation.
- [ ] Set `max_steps: 2000` and the new output root.
- [ ] Hash the copied Source/Target resize implementation and manifests.
- [ ] Verify no checkpoint, adapter, MP4, or recovery state was copied.

### Task 2: Add failing Reference geometry tests

**Files:**
- Create: `tests/test_reference_geometry.py`
- Create: `scripts/reference_geometry.py`

**Interfaces:**
- Produces: `resize_reference_preserving_geometry(path, height, width) -> torch.Tensor`.

- [ ] Write a test asserting a synthetic square Reference returns exactly
  `C,H,W` for all three current model sizes.
- [ ] Write a test asserting a synthetic foreground rectangle retains its
  width/height ratio after conversion.
- [ ] Write a test asserting all-white input is handled deterministically.
- [ ] Run the tests and confirm they fail because the function is absent.

### Task 3: Implement the shared Reference transform

**Files:**
- Implement: `scripts/reference_geometry.py`
- Modify: `scripts/teacher_aligned_dataset.py`
- Modify: `scripts/infer_teacher_aligned_flowfix.py`

**Interfaces:**
- Training and inference both call
  `resize_reference_preserving_geometry(path, height, width)`.

- [ ] Implement foreground-aware crop with white-padding fallback.
- [ ] Replace only `_resize_reference`; do not edit `_resize_video` or
  `target_size`.
- [ ] Route inference through the dataset's shared method.
- [ ] Run the new tests and the existing teacher-aligned contract tests.
- [ ] Compare Source/Target preprocessing hashes with the frozen baseline.

### Task 4: Run data and one-step model preflight

**Files:**
- Create: `preflight/cpu_dataset_report.json`
- Create: `preflight/model_preflight_report.json`

**Interfaces:**
- Consumes: one train sample per native resolution.
- Produces: exact tensor shapes, finite ranges, and one finite training loss.

- [ ] Load one sample from each resolution on CPU.
- [ ] Assert Source, Target, and Reference shapes match.
- [ ] Record transform mode and pixel ranges.
- [ ] Select one idle nonzero physical GPU.
- [ ] Run exactly one optimizer step and require finite loss.

### Task 5: Launch the 2,000-step background training

**Files:**
- Create: `scripts/launch_train.sh`
- Create: `logs/train.log`
- Create: `checkpoints/recovery.ckpt`
- Create: `checkpoints/latest_adapter.safetensors`
- Create: `final_adapter/v2v_lora_and_source_patch.safetensors`

**Interfaces:**
- Produces: a final Scheme-1 adapter and auditable metrics.

- [ ] Add guards rejecting physical GPU 0 and low disk space.
- [ ] Start training detached with the selected nonzero GPU.
- [ ] Save an overwrite recovery checkpoint every 500 steps.
- [ ] Record PID, GPU, config hash, manifest hashes, and source commit/hash.

### Task 6: Monitor and infer

**Files:**
- Create: `validation/logs/*.log`
- Create: `validation/videos/*.mp4`
- Create: `validation/run_manifest.json`

**Interfaces:**
- Consumes: final adapter and three held-out samples.
- Produces: Base/LoRA comparison videos for each native resolution.

- [ ] Create a low-frequency automation that checks process, finite metrics,
  traceback/OOM/NCCL/no-space, GPU binding, disk, and checkpoints.
- [ ] When step 2,000 completes, run the shared-preprocessing inference on a
  nonzero GPU.
- [ ] Save Source, Target, Base, LoRA, and comparison MP4s.
- [ ] Verify frame count, fps, dimensions, duration, and multiple motion frames.

### Task 7: Copy and verify the local handoff

**Files:**
- Create: `D:\Code\Wan2.2_v2v_reference_aspect_scheme1_<timestamp>\**`

**Interfaces:**
- Produces: local scripts, config, reports, adapter, logs, and videos.

- [ ] Copy the complete compact result set from remote to the local folder.
- [ ] Compare SHA256 for the final adapter and every MP4.
- [ ] Write a concise report stating what changed, what stayed fixed, and the
  2,000-step interpretation limit.
