# Wan2.2 V2V LoRA quick validation — 2026-07-13

## Scope and method

- Base model: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B/`
- Training metadata source: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/data/dataset_video_train/metadata_train.csv`
- Isolated experiment directories: `scripts_0713`, `data_0713`, `output_models/wan2.2_lora_quick_v2v_0713`, and `validation_results_0713`.
- The current Wan TI2V pipeline has native V2V inference through `input_video` and `denoising_strength`, but no separate trainable source-video conditioning branch. Therefore this experiment is target-domain LoRA training followed by native input-video restoration testing; it is not paired source-conditioned ControlNet/VACE training.

## Data and training

- 1,024 valid 1280x704 target-domain videos were selected for training.
- Held-out samples `mymotion__002`, `mymotion__012`, and `mymotion__015` were excluded from training.
- Held-out targets and deterministic degraded sources were standardized to 576x320, 15 fps, 33 frames. Exact paths, prompts, degradation recipe, and SHA256 hashes are recorded in `data_0713/v2v_quick_0713/v2v_pairs_0713.csv` and `data_manifest_0713.json`.
- LoRA: DiT `q,k,v,o,ffn.0,ffn.2`, rank 16, learning rate 5e-5, weight decay 0.01, 4 epochs, gradient accumulation 4.
- Eight checkpoints were saved from step 512 through step 4096. Final inference checkpoint: `output_models/wan2.2_lora_quick_v2v_0713/step-4096.safetensors`.
- Final checkpoint SHA256: `8b13c579e6ee2ed5e76c69bbee358511a796e1c18dc3114ced8fd8479660cbc6`.

## Inference

- Identical Base/LoRA settings: 576x320, 33 frames, seed 42, tiled mode, 30 inference steps, denoising strength 0.45.
- Six generated videos are under `validation_results_0713`: `{sample_id}_base.mp4` and `{sample_id}_lora.mp4` for all three held-out samples.
- Four-panel source/target/base/LoRA videos are under `validation_results_0713/comparisons_0713`.

## Aggregate results

| Output | SSIM vs target | PSNR vs target | Sharpness | CLIP prompt similarity |
|---|---:|---:|---:|---:|
| Degraded source | 0.7552 | 21.0526 | 38.2184 | 0.1968 |
| Base V2V | 0.5890 | 18.1502 | 53.9861 | 0.2206 |
| LoRA V2V | 0.6111 | 18.4936 | 45.2508 | 0.2025 |

LoRA versus Base: SSIM +0.0221 (+3.76%) and PSNR +0.3435 dB. SSIM and PSNR improved on all three held-out samples. Sharpness decreased by 16.18%, and CLIP prompt similarity decreased by 8.20%.

## Conclusion

The checkpoint is active and has a repeatable effect: it gives a small, systematic structural-recovery improvement. However, it also softens detail and reduces prompt-semantic similarity. This validates the isolated training/inference pipeline, but the current quick checkpoint should not be presented as a final quality improvement or expanded unchanged into formal training. Next experiments should compare earlier checkpoints, lower LoRA strength, and several denoising strengths. A paper requiring genuinely paired source-conditioned V2V should add an explicit trainable source-video conditioning design.

## Resource status

Training used GPU0 and released it after completion. Final inference used GPU1 and also released it. No `_0713` training or validation process remains running; the experiment is not occupying a GPU.
