# Tiered Formal V2V LoRA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and validate comparable quick, 6,530-step conservative, and 20,000-step recommended paired-V2V LoRA models.

**Architecture:** Reuse the proven 0715 training modules through isolated copies, add exact-step milestone export and a `max_steps=20000` stopping condition, then run the proven deterministic four-panel inference separately for each archived adapter. A Codex heartbeat performs low-cost remote checks and completes the validation/reporting workflow after training.

**Tech Stack:** Python 3.11, PyTorch Lightning DDP, PEFT, safetensors, Wan2.2-TI2V-5B, Decord, ffmpeg, Codex heartbeat automation.

## Global Constraints

- Remote root: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio`.
- New output root: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_tiered_0717`.
- Use physical GPUs 1, 2, and 3; never GPU0; use at most three GPUs.
- Use the frozen 3,916-row paired manifest from `v2v_formal_0715`.
- Keep the 0715 quick adapter unchanged.
- Archive exact steps 6,530 and 20,000 in separate directories.
- Validate all tiers with `mymotion__002`, 33 frames, 576x320, seed 42, 30 steps, and no quality metrics.

---

### Task 1: Isolated configuration and milestone behavior

**Files:**
- Create: `tests/test_tiered_training.py`
- Create: `scripts/train_v2v_lora_tiered.py`
- Create: `configs/train_v2v_lora_tiered_0717.yaml`

**Interfaces:**
- Consumes: the 0715 formal training modules and immutable data manifest.
- Produces: `MilestoneAdapterCheckpoint`, exact milestone output paths, and a trainer stopped by `max_steps`.

- [ ] Write a standard-library unit test that asserts milestones map 6,530 to `conservative/final_adapter` and 20,000 to `recommended/final_adapter`, and that invalid milestone order is rejected.
- [ ] Run the test and confirm it fails because the tiered module does not exist.
- [ ] Copy the proven trainer and minimally add `max_steps` plus immutable milestone export.
- [ ] Run the unit test, syntax compilation, and a two-batch three-GPU preflight; all must exit zero with finite losses.

### Task 2: Launch and low-frequency monitoring

**Files:**
- Create: `scripts/launch_tiered_train.sh`
- Create: `reports/tiered_training.pid`
- Create: `logs/train_tiered_gpu123.log`

**Interfaces:**
- Consumes: Task 1 configuration and training entry point.
- Produces: one background DDP training process and observable metrics/checkpoint state.

- [ ] Validate at least 70,000 MiB free on GPUs 1/2/3, GPU0 exclusion, and at least 10 GiB disk free.
- [ ] Start training with the same Python 3.11 environment and frozen dependency path used in the 0715 run.
- [ ] Confirm one parent and three DDP workers are alive, metrics contain finite loss, and GPU0 has no compute process.
- [ ] Create a 30-minute heartbeat that only checks state and intervenes on evidence of failure or completion.

### Task 3: Archive and validate all three tiers

**Files:**
- Create: `quick/validation/*`
- Create: `conservative/final_adapter/v2v_lora_and_source_patch.safetensors`
- Create: `conservative/validation/*`
- Create: `recommended/final_adapter/v2v_lora_and_source_patch.safetensors`
- Create: `recommended/validation/*`
- Create: `reports/tiered_completion_report.md`

**Interfaces:**
- Consumes: the existing quick adapter and both milestone adapters.
- Produces: three weight paths, three five-video validation packages, and one final comparison report.

- [ ] Copy the already-verified quick visual package into `quick/validation` and record its adapter SHA256.
- [ ] After step 20,000, run the proven inference script once with the conservative adapter and once with the recommended adapter using identical inputs and parameters.
- [ ] Verify every individual video is H.264 576x320 with 33 frames and every comparison video is H.264 1152x640 with 33 frames.
- [ ] Audit adapter tensor counts, hashes, finite values, logs, configs, metrics, and directory inventory; write the final three-tier path report and stop monitoring.

