# Formal V2V comparison design

## Goal

Use the completed formal paired-V2V adapter to generate one deterministic visual comparison without quality metrics.

## Sample and inputs

- Sample: `mymotion__002` from the frozen formal manifest.
- Contract: `input_video + reference_image + caption -> generated target_video`.
- Frames: the first 33 aligned source/target frames.
- Size: 576 x 320 pixels (width x height), matching the training bucket for 1280 x 704 inputs.

## Controlled comparison

Both runs use the same converted Wan2.2-TI2V-5B transformer, native T5 and VAE, source/reference conditioning architecture, initial noise, seed 42, 30 Euler flow-matching steps, and no classifier-free guidance. The Base run uses zero-initialized LoRA layers and the copied pretrained patch embedding for the source branch. The LoRA run additionally loads all 480 trained LoRA tensors and the two trained `patch_embedding_extra` tensors from the final adapter.

## Outputs

The isolated output directory contains individual source, target, Base, and LoRA MP4 files plus a labeled 2x2 comparison MP4. No SSIM, PSNR, LPIPS, or other quality metrics are calculated.

## Safety and validation

Inference must not use GPU0. The script validates manifest paths, adapter key counts, tensor shapes, deterministic noise reuse, finite decoded frames, 33-frame output length, 576 x 320 panel size, and 1152 x 640 comparison size. Logs and a JSON run manifest are retained with the videos.
