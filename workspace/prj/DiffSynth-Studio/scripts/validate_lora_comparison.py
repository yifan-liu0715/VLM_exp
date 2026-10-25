#!/usr/bin/env python
"""
Wan2.2-TI2V-5B LoRA 验证对比脚本
对比原始模型和 LoRA 微调后的效果
"""

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import os

# 配置
MODEL_BASE = "./pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B"
LORA_PATH = "./output_models/wan2.2_lora_quick/epoch-1.safetensors"  # 使用最后一个 epoch
OUTPUT_DIR = "./validation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试样本（从你的数据集中选 3 个代表性的样本）
test_samples = [
    {
        "prompt": "Two women sit in green armchairs indoors, one speaking actively, the other listening with a small white dog between them.",
        "input_image": "./data/dataset_video_train/1280x704/reference/mymotion__002.jpg",
        "output_name": "mymotion_002"
    },
    {
        "prompt": "Two men sit for an interview against a promotional film backdrop, one speaking and gesturing.",
        "input_image": "./data/dataset_video_train/1280x704/reference/mymotion__012.jpg",
        "output_name": "mymotion_012"
    },
    # 可以继续添加更多样本...
]

def load_pipe_with_lora(lora_path=None):
    """加载模型，可选加载 LoRA"""
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=f"{MODEL_BASE}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=[
                f"{MODEL_BASE}/diffusion_pytorch_model-00001-of-00003.safetensors",
                f"{MODEL_BASE}/diffusion_pytorch_model-00002-of-00003.safetensors",
                f"{MODEL_BASE}/diffusion_pytorch_model-00003-of-00003.safetensors"
            ]),
            ModelConfig(path=f"{MODEL_BASE}/Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(path=f"{MODEL_BASE}/google/umt5-xxl/"),
        redirect_common_files=False,
    )
    if lora_path and os.path.exists(lora_path):
        pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
        print(f"✅ 已加载 LoRA: {lora_path}")
    else:
        print("⚠️ 使用原始预训练模型（无 LoRA）")
    return pipe

def generate_video(pipe, sample, suffix=""):
    """生成单个视频"""
    try:
        input_image = Image.open(sample["input_image"]).resize((576, 320))
    except Exception as e:
        print(f"⚠️ 无法加载图片 {sample['input_image']}: {e}")
        return None
    
    video = pipe(
        prompt=sample["prompt"],
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清",
        seed=42,
        height=320,
        width=576,
        input_image=input_image,
        num_frames=33,
        tiled=True,
    )
    output_path = f"{OUTPUT_DIR}/{sample['output_name']}{suffix}.mp4"
    save_video(video, output_path, fps=15, quality=5)
    print(f"✅ 已保存: {output_path}")
    return output_path

def main():
    print("=" * 60)
    print("开始验证对比实验")
    print("=" * 60)
    
    # 1. 加载原始模型（无 LoRA）
    print("\n【1/2】加载原始预训练模型...")
    pipe_base = load_pipe_with_lora(lora_path=None)
    
    # 2. 加载 LoRA 模型
    print("\n【2/2】加载 LoRA 微调模型...")
    pipe_lora = load_pipe_with_lora(lora_path=LORA_PATH)
    
    # 3. 对每个样本生成对比视频
    print("\n开始生成对比视频...")
    print(f"测试样本数: {len(test_samples)}")
    print(f"每个样本生成 2 个视频（原始 vs LoRA）")
    print("=" * 60)
    
    for i, sample in enumerate(test_samples):
        print(f"\n📹 样本 {i+1}/{len(test_samples)}: {sample['output_name']}")
        print(f"   Prompt: {sample['prompt'][:60]}...")
        
        # 原始模型生成
        print("   🔹 生成原始模型视频...")
        generate_video(pipe_base, sample, suffix="_base")
        
        # LoRA 模型生成
        print("   🔸 生成 LoRA 模型视频...")
        generate_video(pipe_lora, sample, suffix="_lora")
    
    print("\n" + "=" * 60)
    print("✅ 验证完成！")
    print(f"所有视频保存在: {OUTPUT_DIR}/")
    print("对比规则: *_base.mp4 = 原始模型, *_lora.mp4 = LoRA 微调模型")
    print("=" * 60)

if __name__ == "__main__":
    main()
