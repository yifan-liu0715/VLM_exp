# Wan V2V Teacher-Aligned Recommended Experiment Design

## Objective

Create one isolated 20,000-step Wan2.2-TI2V-5B V2V LoRA experiment that follows the teacher-provided EffectV2V reference as closely as the current three-GPU shared server permits. The experiment must make the differences from the earlier `v2v_corrected_tiers_0719` run explicit so that visual results can help identify whether the earlier blur/artifacts came from the model/loss path, training settings, inference settings, or data evaluation leakage.

## Authoritative references

- Teacher archive on the local workstation:
  `C:\Users\86173\Desktop\暂存\AAAI\Classin老师资源\EffectV2V.tar`
- Extracted teacher source:
  `C:\Users\86173\Desktop\暂存\AAAI\Classin老师资源\EffectV2V\StreamingEffect`
- Remote project:
  `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio`
- New isolated remote root:
  `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_teacher_aligned_recommended_0724`
- Existing corrected manifest:
  `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_corrected_tiers_0719/metadata/metadata_corrected_direction_0719.csv`
- HF-format Wan components:
  `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/pretrain_weights/Wan2.2-TI2V-5B-HF`

## Experiment contract

The semantic data direction is:

```text
physical output_video (source A)
  + external reference image (reference B)
  + caption describing target B
  -> physical input_video (target B)
```

The new manifest adapter must expose this direction without renaming or copying the original media files. Only rows whose `source_split` equals `train` may be presented to the training DataLoader. Validation and test rows must remain available only for inference and audit.

## Architecture

The experiment will vendor a frozen copy of the teacher EffectV2V source files and record their SHA-256 hashes. The teacher transformer, condition attention processor, Flow Matching scheduler, loss path, LoRA target modules, source-condition patch, and UniPC inference algorithm remain unchanged.

A small, isolated dataset adapter will translate the existing paired manifest into the batch dictionary expected by `wan2_trainer_plus.py`:

- `pixel_values`: source video latent input.
- `pixel_values2`: target video to be noised and predicted.
- `first_frames`: external reference image when retained.
- `prompts`: target caption.

The adapter will sample aligned 97-frame clips from paired source/target videos, apply bucket resizing, and support reference dropout. Text dropout remains in the teacher trainer.

The server does not contain the complete `Wan-AI/Wan2.2-TI2V-5B-Diffusers` repository expected by the teacher loader. Both available Wan directories store the text encoder and VAE as original `.pth` files rather than Diffusers `text_encoder/` and `vae/` subdirectories, while only the transformer has already been converted. The server also cannot reach Hugging Face, and downloading another complete model is incompatible with the remaining disk budget.

Therefore, a thin system subclass will reuse the same local DiffSynth text-encoder and VAE loading path as the previous run while keeping the teacher forward/loss and optimization path. This controlled exception is mandatory and must be recorded in `experiment_manifest.json`; the experiment must be described as **teacher-aligned**, not an exact execution of the teacher environment. Keeping this loading path fixed also makes the comparison more diagnostic: changes in artifacts can be attributed to the teacher-aligned training and inference recipe rather than to replacing the VAE and text encoder at the same time.

## Training settings

The configuration follows `configs/train_teacher.yaml` except for explicit hardware and experiment requirements:

| Setting | Value |
|---|---:|
| Maximum steps | 20,000 |
| Formal output tier | `recommended` only |
| Frames | 97 |
| FPS metadata | 24 |
| LoRA rank / alpha | 96 / 96 |
| LoRA dropout | 0 |
| Base learning rate | `1e-4` |
| AdamW betas | `(0.9, 0.95)` |
| Weight decay | `1e-2` |
| Warmup | 2 optimizer steps, then constant multiplier |
| Flow shift | 5 |
| Precision | `bf16-mixed` |
| Gradient checkpointing | enabled |
| Gradient clipping | disabled |
| Text dropout | 0.1 |
| Reference dropout | 0.5 |
| Reference source | external reference first; random target frame only as fallback |
| Seed | 1234 |

The teacher bucket policy (align to 32, long side 1088) is tested first. The measured preflight at 97 frames and gradient accumulation 4 completed one optimizer step in about 167 seconds, which projects to approximately 39 days for 20,000 steps on the currently available GPU. The formal comparison therefore uses the previously authorized aspect-ratio-preserving reduction to long side 704. The successful 1088 preflight and the selected 704 setting must both remain in the experiment evidence.

The teacher per-device batch size 4 is not practical at 97 frames on this shared server. The formal run uses per-device batch size 1 and, matching the teacher YAML, `accumulate_grad_batches=1`. The final effective global batch is recorded from the actual number of GPUs used. This is a documented hardware adaptation rather than an exact reproduction of the teacher global batch.

## Environment policy

The first preflight uses the already working `diffsynth_fixed` environment, but it must record all relevant package versions. If the vendored teacher transformer/attention code fails because of an API/version incompatibility, create an isolated environment from the teacher `environment.yaml` rather than modifying an existing shared environment. The local DiffSynth VAE/text loader exception remains in either environment.

The `v2v_train` environment is not currently usable for the teacher trainer because it lacks `pytorch_lightning`, `omegaconf`, and `decord`. It must not be mutated in place.

## GPU and storage policy

- Physical GPU0 is forbidden at every stage.
- Use at most three unique nonzero physical GPUs.
- Select GPUs only after a fresh `nvidia-smi` check.
- A launcher must reject empty `CUDA_VISIBLE_DEVICES`, GPU0, duplicate IDs, and more than three IDs.
- Do not copy the dataset or pretrained weights.
- The formal run saves one overwrite-style recovery adapter and one immutable final recommended adapter.
- Full Lightning model checkpoints are disabled to protect the approximately 70 GB remaining on `/nvmedata`.

## Checkpoint and output layout

```text
v2v_teacher_aligned_recommended_0724/
  teacher_reference/
  configs/
    train_teacher_aligned_20000.yaml
    resolved_train_config.yaml
  metadata/
    train_manifest.csv
    val_manifest.csv
    test_manifest.csv
    manifest_report.json
  scripts/
  tests/
  preflight/
  logs/
  checkpoints/
    latest_adapter.safetensors
  recommended/
    final_adapter/
      v2v_lora_and_source_patch.safetensors
    validation/
      videos/
      reports/
  experiment_manifest.json
```

No `quick` or `conservative` output directory is created.

## Inference and comparison

After training reaches exactly 20,000 optimizer steps:

- Use the teacher UniPC scheduler with flow shift 5.
- Use 50 inference steps.
- Use CFG 5.
- Keep source and reference conditions in both conditional and unconditional passes, matching the teacher pipeline.
- Use seed 42 for direct visual comparison with the previous run.
- Generate base and new-adapter videos with identical inputs and sampling parameters.
- Generate at least one seen diagnostic sample (`mymotion__002`) and at least one held-out test sample.

The old `v2v_corrected_tiers_0719` adapter trained on all splits, so its held-out comparison is contaminated. Reports must label the old result as diagnostic only and must not claim a fair generalization comparison.

## Verification gates

Formal training may start only if all of the following pass:

1. The isolated root does not reuse old output paths.
2. Every train row has `source_split=train`; no val/test path appears in the train manifest.
3. A known sample resolves source, target, reference, and caption in the documented direction.
4. Source and target decode at least 97 aligned frames.
5. Teacher reference hashes match the local archive.
6. Transformer and condition-attention components load from the converted HF-format transformer, while the VAE and text encoder load from the documented original `.pth` components.
7. Trainable parameters consist only of attention LoRA weights and `patch_embedding_extra`.
8. The loss is finite for at least one forward/backward optimizer step.
9. The preflight process and formal launcher exclude physical GPU0.
10. Resolved configuration, environment versions, selected GPU IDs, effective batch, and spatial cap are persisted before the formal run.

## Diagnostic interpretation

This run changes several earlier mismatches together in order to answer whether a teacher-aligned recipe removes the severe artifacts. It is a recipe-level comparison, not a one-variable ablation.

- If the new adapter is substantially sharper with the teacher-aligned UniPC/CFG path, the earlier scheduler/dropout/optimization mismatch is the leading explanation.
- If the new adapter remains severely blurred on seen and held-out samples, investigate VAE latent compatibility and the source/reference conditioning architecture before adding perceptual losses.
- If seen samples improve but held-out samples fail, the dominant issue is data size/diversity or overfitting rather than the Flow Matching equation.
- A falling latent MSE alone is not evidence of perceptual improvement.
