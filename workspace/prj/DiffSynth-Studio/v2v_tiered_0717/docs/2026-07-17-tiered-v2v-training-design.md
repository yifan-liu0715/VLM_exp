# Tiered formal V2V LoRA training design

## Goal

Produce directly comparable quick, conservative, and recommended Wan2.2 paired-V2V LoRA checkpoints and a four-panel visual validation for each tier.

## Training trajectory

The existing 5-epoch formal result at step 3,265 is the quick tier. One new run starts from the same converted Wan2.2-TI2V-5B base, frozen paired manifest, seed 42, rank-64 LoRA, optimizer, three-GPU effective batch size, and source-video condition branch. It archives step 6,530 as the conservative tier and continues without resetting optimizer or scheduler to step 20,000 as the recommended tier. This avoids recomputing an identical first 6,530-step trajectory and makes training duration the only intended variable.

## Isolation

All new files live under `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_tiered_0717`. Shared scripts, configuration, logs, and monitoring state are separate from `v2v_formal_0715`. Conservative and recommended adapters and validation results occupy distinct subdirectories. The existing quick adapter remains authoritative at its 0715 path and its validation videos are copied into the tiered report tree for a uniform comparison package.

## Validation

All three adapters use the same `mymotion__002` source, target, reference image, prompt, first 33 frames, 576x320 resolution, seed 42, and 30 denoising steps. Each tier exports source, target, Base, LoRA, and labeled 2x2 comparison MP4 files. No quality metrics are computed in this phase.

## Runtime and monitoring

Training uses physical GPUs 1, 2, and 3 only; GPU0 is forbidden. A compact latest adapter overwrites every 500 steps, while immutable milestone adapters are written at 6,530 and 20,000. A low-frequency Codex heartbeat checks training liveness, finite loss, traceback/OOM/NCCL/no-space errors, GPU0 exclusion, disk space, milestone presence, and final validation. It does not run inference while training is healthy and stops itself after all three validation packages are audited.

