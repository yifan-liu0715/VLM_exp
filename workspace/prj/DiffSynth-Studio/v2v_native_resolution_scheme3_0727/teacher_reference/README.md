# EffectV2V

A reference-conditioned video effect generation framework based on bidirectional attention. Given a source video and a reference image or text prompt, the model applies expressive visual effects while preserving human identity, background content, and temporal consistency.

---

## Overview

**EffectV2V** uses a bidirectional transformer trained with LoRA on reference-conditioned in-context video editing. The model supports:

- **Reference-conditioned control**: inject a keyframe image to guide the effect style
- **Text-driven control**: describe effects with natural language
- Built on [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)

---

## Installation

```bash
# Create conda environment
conda create -n effectv2v python=3.10
conda activate effectv2v

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install diffusers transformers accelerate
pip install decord pillow numpy
pip install einops omegaconf
pip install pytorch-lightning wandb
```

---

## Inference

```bash
# Set CKPT path in infer_teacher.sh, then:
CKPT=./checkpoints/model.ckpt bash infer_teacher.sh
```

Or directly:

```bash
python src/wan2_inference_testset.py \
    --config configs/infer_teacher.yaml \
    --ckpt_path ./checkpoints/model.ckpt \
    --testset /path/to/testset \
    --output_dir ./outputs
```

**Test set format:** Each sample consists of `{stem}.mp4` (first half = source video, second half = GT), `{stem}.txt` (text prompt), and optionally `{stem}.png` (reference effect image).

---

## Training

### Prerequisites

1. Download the Wan2.2-TI2V-5B-Diffusers backbone:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./models/Wan2.2-TI2V-5B-Diffusers
```

2. Prepare your dataset and set `dataset_roots` in `configs/train_teacher.yaml`.

3. (Optional) Precompute latents for faster training:

```bash
python scripts/precompute_v2v_bg_latents.py \
    --config configs/train_teacher.yaml
```

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_teacher.sh
```

Key hyperparameters (edit `configs/train_teacher.yaml`):
- `training.max_steps`: 8000
- `training.learning_rate`: 1e-4
- `dataset.max_long_side`: 1088

### Post-training: Merge LoRA

After training, merge the LoRA weights into a single checkpoint:

```bash
python scripts/merge_v2v_bg_lora.py \
    --ckpt_path ./checkpoints/model.ckpt \
    --output_path ./checkpoints/merged_transformer.pt
```

---

## Acknowledgements

This project builds on [Wan2.2](https://github.com/Wan-Video/Wan2.1) by Alibaba and [FastGen](https://github.com/NVlabs/FastGen) by NVIDIA. We thank the authors for open-sourcing their work.
