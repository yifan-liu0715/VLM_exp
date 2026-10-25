# Corrected V2V Tiered Training Implementation Plan

> **For agentic workers:** Execute inline with test-first checkpoints; this directory is deliberately isolated from all 0713/0715 artifacts.

**Goal:** Train one corrected-direction Wan2.2 V2V LoRA trajectory and persist quick, conservative, and recommended adapters, then generate one identical four-panel comparison for each tier.

**Architecture:** A frozen manifest swaps the earlier source/target columns so the current `output_video` becomes the driving source and the current `input_video` becomes ground truth. One DDP run on three explicitly nonzero physical GPUs saves immutable adapters at steps 3265, 6530, and 20000. A controller starts inference only after training succeeds and uses one idle nonzero GPU sequentially for all three tiers.

**Tech Stack:** Python 3.11, PyTorch Lightning DDP, DiffSynth-Studio/Wan2.2, PEFT LoRA, safetensors, Bash, ffmpeg.

## Global Constraints

- Remote root: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_corrected_tiers_0719`.
- Correct direction: existing `output_video` + `reference_image` + existing caption -> existing `input_video`.
- GPU0 is forbidden in training, inference, preflight, and monitoring.
- Use at most three GPUs; launch default is physical GPUs 1, 2, and 4.
- Training resolution remains bucket-scaled: 640x256, 576x320, or 320x576; 33 frames.
- LoRA rank/alpha/dropout: 64/64/0; learning rate 1e-4; BF16; accumulation 2; seed 42.
- Save milestones at step 3265 (quick), 6530 (conservative), and 20000 (recommended).
- Generate no quality metrics; output one source/target/base/LoRA four-panel comparison for `mymotion__002` per tier.

---

### Task 1: Freeze the corrected manifest contract

**Files:**
- Create: `tests/test_corrected_tiered_training.py`
- Create: `scripts/tiered_training_utils.py`
- Create: `scripts/build_corrected_manifest.py`

- [ ] Write tests proving only `input_video` and `output_video` are swapped, while reference and caption remain unchanged.
- [ ] Run tests and confirm failure because the utility module does not exist.
- [ ] Implement the minimal swap and milestone validation helpers.
- [ ] Run tests and confirm all pass.
- [ ] Build the 3916-row manifest, validate every media path, and save a SHA-256 report.

### Task 2: Implement continuous milestone training

**Files:**
- Create: `configs/train_v2v_corrected_tiers_0719.yaml`
- Create: `scripts/train_v2v_lora_tiered.py`
- Create: `scripts/launch_tiered_train.sh`

- [ ] Extend the proven 0715 Lightning system without changing its V2V loss/model architecture.
- [ ] Add an atomic milestone callback that writes one adapter and JSON record at each exact global step.
- [ ] Set `max_steps=20000` and preserve a rolling checkpoint every 500 steps.
- [ ] Reject GPU0, more than three GPUs, GPUs with under 70 GiB free, and insufficient disk before model loading.
- [ ] Compile scripts and run a one-optimizer-step DDP preflight on three nonzero GPUs.

### Task 3: Chain deterministic inference

**Files:**
- Create: `scripts/run_train_then_validate.sh`
- Copy: `scripts/infer_v2v_comparison.py`

- [ ] Wait for successful training termination and verify all three adapters.
- [ ] Choose an idle physical GPU from the allowed nonzero set.
- [ ] Run the proven 33-frame, 30-step, seed-42 comparison for the same sample and config at every tier.
- [ ] Verify each output with ffprobe and write controller state/log files.

### Task 4: Launch and monitor

**Files:**
- Create: `logs/controller.log`
- Create: `reports/controller.pid`
- Create: `reports/controller_state.json`

- [ ] Start the controller with `nohup` so it survives the SSH session.
- [ ] Confirm the controller and three DDP ranks are alive and GPU0 has no process from this run.
- [ ] Create a 30-minute low-cost monitor for process, finite loss, error signatures, GPU0, disk, adapters, and final MP4s.
- [ ] On completion, audit hashes/config/manifest/logs/videos and stop the monitor.
