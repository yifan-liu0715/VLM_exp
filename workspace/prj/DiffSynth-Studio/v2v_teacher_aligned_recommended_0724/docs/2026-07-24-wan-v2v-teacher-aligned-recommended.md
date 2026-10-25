# Wan V2V Teacher-Aligned Recommended Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, preflight, and launch one isolated 20,000-step teacher-aligned Wan2.2 V2V LoRA recommended experiment, then generate auditable base-versus-adapter comparison videos.

**Architecture:** Vendor the teacher EffectV2V reference for traceability, but use a thin local system implementation because the server lacks the complete Diffusers VAE/text-encoder layout required by the teacher loader. Reuse the converted teacher transformer and condition attention code, adapt the corrected paired manifest into 97-frame batches, and reproduce the teacher Flow Matching objective, LoRA settings, dropout, optimizer, LR scaling, and UniPC inference. All artifacts live under one new remote root and no dataset or base weight is copied.

**Tech Stack:** Python 3.11, PyTorch 2.6, PyTorch Lightning 2.6, Diffusers, PEFT, OmegaConf, Decord, safetensors, Wan2.2-TI2V-5B, Bash/SSH.

## Global Constraints

- Remote root: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_teacher_aligned_recommended_0724`.
- Formal training must stop at exactly 20,000 optimizer steps.
- Create only the `recommended` formal tier; do not create quick or conservative tiers.
- Physical GPU0 is forbidden; use at most three unique nonzero GPUs.
- Use only `source_split=train` rows for training.
- Data direction is physical `output_video` + reference + target caption -> physical `input_video`.
- Teacher-aligned defaults: 97 frames, rank/alpha 96/96, text dropout 0.1, reference dropout 0.5, AdamW betas 0.9/0.95, weight decay 0.01, two-step warmup then constant LR, flow shift 5, seed 1234.
- Use batch size 1 and the teacher YAML's gradient accumulation 1.
- Retain the successful 1088 preflight as evidence; use max long side 704 formally because measured 1088 throughput projects to roughly 39 days for 20,000 steps.
- Do not copy datasets or base weights.
- Disable full Lightning checkpoints; retain one overwrite-style recovery adapter and one immutable final adapter.

---

### Task 1: Freeze teacher reference and split manifests

**Files:**
- Create: `teacher_aligned_0724_staging/scripts/build_teacher_aligned_manifests.py`
- Create: `teacher_aligned_0724_staging/tests/test_manifest_contract.py`
- Copy: teacher `models/wan2`, `tools/my_schedule.py`, trainer, pipeline, YAML files, `environment.yaml`, and `requirements.txt` into `teacher_aligned_0724_staging/teacher_reference/`

**Interfaces:**
- Consumes: corrected CSV at `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_corrected_tiers_0719/metadata/metadata_corrected_direction_0719.csv`.
- Produces: `build_manifests(source_csv: Path, output_dir: Path) -> dict`, split CSVs, hash report, and a known-sample direction audit.

- [ ] **Step 1: Write the failing manifest tests**

```python
def test_train_manifest_contains_only_train_rows(tmp_path):
    report = build_manifests(FIXTURE, tmp_path)
    rows = list(csv.DictReader((tmp_path / "train_manifest.csv").open()))
    assert rows
    assert {row["source_split"] for row in rows} == {"train"}
    assert report["contract"] == "physical output_video + reference + caption -> physical input_video"


def test_known_sample_direction(tmp_path):
    report = build_manifests(FIXTURE, tmp_path)
    sample = report["known_samples"]["mymotion__002"]
    assert "/output_video/" in sample["source_video"]
    assert "/input_video/" in sample["target_video"]
```

- [ ] **Step 2: Run the tests and confirm they fail before implementation**

Run:

```powershell
python -m pytest teacher_aligned_0724_staging/tests/test_manifest_contract.py -q
```

Expected: import failure because `build_teacher_aligned_manifests.py` does not exist.

- [ ] **Step 3: Implement deterministic split generation**

The implementation must:

```python
def build_manifests(source_csv: Path, output_dir: Path) -> dict:
    rows = list(csv.DictReader(source_csv.open(encoding="utf-8-sig")))
    split_rows = {name: [] for name in ("train", "val", "test")}
    for row in rows:
        split = row["source_split"].strip()
        if split not in split_rows:
            raise ValueError(f"unsupported source_split={split!r}")
        split_rows[split].append(row)
    # Preserve corrected manifest columns:
    # row["input_video"] is source physical output_video.
    # row["output_video"] is target physical input_video.
    ...
```

It must not copy or rename media.

- [ ] **Step 4: Freeze and hash the teacher files**

Copy only the required reference files to local staging, create `teacher_reference_sha256.json`, and verify the hashes against the source archive.

- [ ] **Step 5: Run manifest tests**

Expected: all tests pass, row counts equal train=2993, val=468, test=455, and `mymotion__002` resolves in the documented direction.

### Task 2: Implement the 97-frame teacher-aligned dataset

**Files:**
- Create: `teacher_aligned_0724_staging/scripts/teacher_aligned_dataset.py`
- Create: `teacher_aligned_0724_staging/tests/test_teacher_aligned_dataset.py`

**Interfaces:**
- Consumes: one split manifest, `sample_frames`, `max_long_side`, `bucket_align`.
- Produces: `TeacherAlignedPairedDataset.__getitem__()` with `source_video`, `target_video`, `reference_image`, `prompt`, `sample_id`, and `bucket`.

- [ ] **Step 1: Write dataset contract tests**

```python
def test_aligned_indices_are_identical_for_source_and_target():
    indices = aligned_frame_indices(paired_length=118, sample_frames=97, start=7)
    assert indices[0] == 7
    assert indices[-1] == 103
    assert len(indices) == 97


def test_bucket_size_preserves_aspect_ratio_and_alignment():
    assert target_size(native_h=704, native_w=1280, max_long_side=1088, align=32) == (576, 1088)
```

- [ ] **Step 2: Confirm tests fail**

Run:

```powershell
python -m pytest teacher_aligned_0724_staging/tests/test_teacher_aligned_dataset.py -q
```

- [ ] **Step 3: Implement aligned video/reference loading**

Use the same randomly selected start index for source and target, decode exactly 97 frames, normalize RGB to `[-1, 1]`, and resize all three conditions to the same bucket size. Return the external reference without applying dropout in the dataset; dropout is applied batch-wise in the system to avoid mixed-key collation failures.

- [ ] **Step 4: Verify a real known sample on the remote CPU**

Run the dataset against `mymotion__002` and assert:

```text
source_video == target_video shape == [3, 97, H, W]
reference_image shape == [3, H, W]
source path contains /output_video/
target path contains /input_video/
```

### Task 3: Implement teacher-aligned model, objective, and optimizer

**Files:**
- Create: `teacher_aligned_0724_staging/scripts/teacher_aligned_system.py`
- Create: `teacher_aligned_0724_staging/tests/test_teacher_alignment_contract.py`

**Interfaces:**
- Consumes: resolved OmegaConf configuration and batches from `TeacherAlignedPairedDataset`.
- Produces: `TeacherAlignedV2VSystem`, finite weighted Flow Matching loss, teacher-scaled AdamW, and compact adapter export.

- [ ] **Step 1: Write static and unit contract tests**

Tests must assert:

```python
assert config.lora.rank == 96
assert config.lora.alpha == 96
assert config.training.text_drop_prob == 0.1
assert config.training.reference_drop_prob == 0.5
assert config.optimizer.beta2 == 0.95
assert config.optimizer.warmup_steps == 2
assert config.training.num_frames == 97
```

The source of `training_step` must contain:

```text
hidden_states=noisy_target
encoder_contion_states=source
encoder_first_states=reference
velocity_target = noise - target
```

- [ ] **Step 2: Confirm contract tests fail**

- [ ] **Step 3: Implement the model loader**

Load:

```text
tokenizer: pretrain_weights/Wan2.2-TI2V-5B-HF/google/umt5-xxl
text encoder: pretrain_weights/Wan2.2-TI2V-5B-HF/models_t5_umt5-xxl-enc-bf16.pth
VAE: pretrain_weights/Wan2.2-TI2V-5B-HF/Wan2.2_VAE.pth
transformer: v2v_formal_0715/base_components/transformer
```

Install the vendored teacher `ConditionAttnProcessor2_0`, add rank-96 LoRA to `to_q`, `to_k`, `to_v`, and `to_out.0`, and train `patch_embedding_extra`.

- [ ] **Step 4: Implement batch-level teacher dropout and objective**

```python
prompts = [p if random.random() > 0.1 else "" for p in prompts]
drop_reference = random.random() < 0.5
reference = torch.zeros_like(encoded_reference) if drop_reference else encoded_reference

noise = torch.randn_like(target)
noisy_target = scheduler.add_noise(target, noise, timestep)
velocity_target = scheduler.training_target(target, noise, timestep)
prediction = transformer(
    hidden_states=noisy_target,
    encoder_hidden_states=prompt_embeddings,
    timestep=timestep_sequence,
    attention_kwargs={
        "encoder_contion_states": source,
        "encoder_first_states": reference,
    },
    return_dict=False,
)[0]
loss = weighted_mse(prediction, velocity_target)
```

- [ ] **Step 5: Implement teacher optimizer behavior**

The actual LR is:

```python
scaled_lr = base_lr * math.sqrt(accumulate_grad_batches * num_gpus * num_nodes)
```

Use AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.01`, and a two-step warmup followed by a constant multiplier.

- [ ] **Step 6: Implement compact atomic checkpoints**

Export only LoRA tensors and `patch_embedding_extra`, atomically replacing `checkpoints/latest_adapter.safetensors`; at global step 20,000 write `recommended/final_adapter/v2v_lora_and_source_patch.safetensors`.

### Task 4: Configuration, launcher, and preflight gates

**Files:**
- Create: `teacher_aligned_0724_staging/configs/train_teacher_aligned_20000.yaml`
- Create: `teacher_aligned_0724_staging/scripts/train_teacher_aligned.py`
- Create: `teacher_aligned_0724_staging/scripts/preflight.py`
- Create: `teacher_aligned_0724_staging/scripts/launch_train.sh`
- Create: `teacher_aligned_0724_staging/tests/test_config_and_gpu_guards.py`

**Interfaces:**
- Consumes: nonzero `CUDA_VISIBLE_DEVICES` and the resolved YAML.
- Produces: preflight report, formal log, PID file, resolved config, environment manifest, and final adapter.

- [ ] **Step 1: Write GPU guard tests**

```python
@pytest.mark.parametrize("value", ["", "0", "0,2", "2,2", "1,2,4,6"])
def test_invalid_gpu_lists_are_rejected(value):
    with pytest.raises(RuntimeError):
        explicit_nonzero_visible_gpus(value)


def test_three_nonzero_gpus_are_allowed():
    assert explicit_nonzero_visible_gpus("1,2,4") == ["1", "2", "4"]
```

- [ ] **Step 2: Implement config and launch validation**

Formal startup must fail unless:

```text
max_steps=20000
milestone outputs are absent
num_frames=97
rank=alpha=96
train manifest contains train rows only
disk free space >= 20 GiB
GPU0 absent
```

- [ ] **Step 3: Implement one-batch preflight mode**

Preflight runs one optimizer step and writes:

```json
{
  "status": "passed",
  "loss_is_finite": true,
  "num_frames": 97,
  "max_long_side": 1088,
  "physical_gpus": ["2"],
  "trainable_tensor_contract": {
    "lora": 480,
    "source_patch": 2,
    "other": 0
  }
}
```

If and only if the log contains a genuine CUDA OOM, retry 704 then 640, clearing the failed preflight process before retrying.

- [ ] **Step 4: Implement formal background launch**

Use `nohup` and persist controller PID. The launcher must not choose GPU0 and must not start if the selected GPU has an unrelated compute process that leaves insufficient memory for the successful preflight configuration.

### Task 5: Deploy and run remote preflight

**Files:**
- Deploy the local staging tree to the new remote root.

**Interfaces:**
- Consumes: verified local staging files.
- Produces: a remote isolated experiment ready for formal training.

- [ ] **Step 1: Run all local pure-Python tests**

Run:

```powershell
python -m pytest teacher_aligned_0724_staging/tests -q
```

Expected: zero failures.

- [ ] **Step 2: Create the remote root and copy staging files**

Use `scp`/`rsync` without overwriting any earlier experiment root.

- [ ] **Step 3: Re-run syntax and pure contract tests remotely**

Use `/nvmedata/workspace2/conda_data/envs/diffsynth_fixed/bin/python`.

- [ ] **Step 4: Build split manifests remotely and verify counts**

Expected:

```text
train=2993
val=468
test=455
```

- [ ] **Step 5: Select a currently free nonzero GPU and run the one-step preflight**

Capture complete stdout/stderr and GPU selection evidence.

- [ ] **Step 6: Audit the preflight report**

Do not launch formal training if loss is non-finite, trainable tensors differ from LoRA plus source patch, source/target direction fails, or GPU0 appears anywhere in the resolved launch state.

### Task 6: Launch and monitor the 20,000-step recommended run

**Files:**
- Runtime outputs under the new remote root only.

**Interfaces:**
- Consumes: successful preflight configuration.
- Produces: final recommended adapter and complete training evidence.

- [ ] **Step 1: Start the formal process on the successful nonzero GPU set**

- [ ] **Step 2: Verify early progress**

Confirm controller/ranks are alive, global step advances, loss is finite, and no traceback/OOM/NCCL/no-space error appears.

- [ ] **Step 3: Switch to low-frequency monitoring**

Check only process liveness, latest metric, disk, GPU0 exclusion, recovery adapter, and fatal error patterns. Do not run inference while training is healthy.

- [ ] **Step 4: Audit completion at exactly 20,000 steps**

Verify:

```text
final metrics step = 19999 or trainer global_step = 20000
recommended final adapter exists
adapter tensor contract = 480 LoRA + 2 source patch tensors
loss values are finite
no train/val/test leakage in the train manifest
```

### Task 7: Teacher-aligned inference and comparison

**Files:**
- Create: `teacher_aligned_0724_staging/scripts/infer_teacher_aligned.py`
- Create: `teacher_aligned_0724_staging/tests/test_inference_contract.py`

**Interfaces:**
- Consumes: final adapter, source/reference/prompt rows, seed 42.
- Produces: source, target, base, adapter, and four-panel MP4s plus per-run manifest.

- [ ] **Step 1: Write inference contract tests**

Assert UniPC, flow shift 5, 50 steps, CFG 5, identical base/adapter seed and conditions, and both conditional/unconditional passes retaining source/reference.

- [ ] **Step 2: Implement inference using the vendored teacher algorithm**

- [ ] **Step 3: Generate one seen and one held-out comparison on a nonzero GPU**

- [ ] **Step 4: Run `ffprobe` and inspect multiple motion frames**

- [ ] **Step 5: Write the diagnostic comparison report**

The report must distinguish:

```text
old adapter: trained on all splits, diagnostic only
new adapter: trained on train split only
base: untrained V2V condition interface
```

It must state whether artifacts improved, persisted, or changed, without treating latent training loss as a perceptual metric.

## Plan self-review

- Every explicit requirement in the design is covered by a task.
- No task creates quick or conservative outputs.
- The unavailable full Diffusers VAE/text layout is recorded as a controlled exception.
- The plan contains no dataset or base-weight copy step.
- Formal training is gated by one finite optimizer step and explicit GPU0 exclusion.
- Completion requires both a 20,000-step adapter audit and visual inference artifacts; merely launching training is not completion.
