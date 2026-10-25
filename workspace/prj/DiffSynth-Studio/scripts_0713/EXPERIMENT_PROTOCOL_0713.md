# Wan2.2-TI2V-5B native V2V + LoRA experiment (_0713)

## Scope

This is an isolated V2V restoration smoke test using the repository's native `input_video` path. It does not modify or overwrite the earlier TI2V experiment.

The plain Wan2.2-TI2V-5B pipeline supports V2V inference by encoding `input_video`, adding noise according to `denoising_strength`, and denoising with the same DiT. It does not contain a separate trainable source-video control branch. Therefore:

- LoRA training learns the held-out-disjoint target video domain from target videos and prompts.
- V2V evaluation uses recorded source/target pairs. Source videos enter inference through `input_video`.
- This must not be described as a paired source-conditioned ControlNet/VACE training experiment.

## Isolated paths

```text
scripts_0713/
data_0713/v2v_quick_0713/
output_models/wan2.2_lora_quick_v2v_0713/
validation_results_0713/
```

## Data

- Training: first 1024 lexicographically sorted valid `1280x704` rows from `metadata_train.csv`, excluding `mymotion__002`, `mymotion__012`, and `mymotion__015`.
- Training target resolution: 576x320, 33 frames.
- Held-out V2V pairs: `mymotion__002`, `mymotion__012`, `mymotion__015`.
- Source: deterministic degraded 576x320/33-frame version of the held-out original.
- Target: standardized high-quality 576x320/33-frame version of the same held-out original.

Every selected training ID, source/target absolute path, degradation recipe, and source/target SHA-256 is recorded under `data_0713/v2v_quick_0713/`.

## Training

- Base: Wan2.2-TI2V-5B.
- LoRA: DiT `q,k,v,o,ffn.0,ffn.2`, rank 16.
- LR: 5e-5; weight decay: 0.01.
- Epochs: 4; gradient accumulation: 4.
- Save every 512 data steps.

## V2V inference

- Same `source_video`, prompt, seed=42, 576x320, 33 frames, tiled=True.
- `denoising_strength=0.45`, 30 inference steps.
- Only difference: base versus the selected `_0713` LoRA checkpoint.

## Metrics

- Framewise SSIM to target.
- PSNR to target.
- Laplacian sharpness.
- CLIP ViT-B/32 prompt-video similarity when the cached model and Python package are available.
