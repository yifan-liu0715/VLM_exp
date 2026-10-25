#!/usr/bin/env python
"""
Wan2.2-TI2V-5B Video-to-Video LoRA 训练脚本
用法: python scripts/train_v2v.py
"""

import os
import subprocess
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def main():
    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    
    # 构建训练命令
    cmd = [
        "python", "examples/wanvideo/model_training/train.py",
        "--task", "sft",
        "--dataset_base_path", "./data/dataset_video_train",
        "--dataset_metadata_path", "./data/dataset_video_train/metadata_train.csv",
        "--data_file_keys", "video,reference_image",
        "--model_paths", '[\n'
        '        "./pretrain_weights/Wan2.2-TI2V-5B-HF/models_t5_umt5-xxl-enc-bf16.pth",\n'
        '        [\n'
        '            "./pretrain_weights/Wan2.2-TI2V-5B-HF/diffusion_pytorch_model-00001-of-00003.safetensors",\n'
        '            "./pretrain_weights/Wan2.2-TI2V-5B-HF/diffusion_pytorch_model-00002-of-00003.safetensors",\n'
        '            "./pretrain_weights/Wan2.2-TI2V-5B-HF/diffusion_pytorch_model-00003-of-00003.safetensors"\n'
        '        ],\n'
        '        "./pretrain_weights/Wan2.2-TI2V-5B-HF/Wan2.2_VAE.pth"\n'
        '    ]',
        "--tokenizer_path", "./pretrain_weights/Wan2.2-TI2V-5B-HF/google/umt5-xxl/",
        "--extra_inputs", "reference_image",
        "--trainable_models", "dit",
        "--lora_base_model", "dit",
        "--lora_target_modules", "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank", "32",
        "--output_path", "./output_models/wan2.2_lora_test",
        "--num_epochs", "3",
        "--gradient_accumulation_steps", "4",
        "--use_gradient_checkpointing",
        "--learning_rate", "1e-4"
    ]
    
    # 执行命令（使用 GPU 1）
    cmd = ["CUDA_VISIBLE_DEVICES=1"] + cmd
    cmd_str = " ".join(cmd)
    
    print("=" * 60)
    print("启动 Wan2.2-TI2V-5B Video-to-Video 训练")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"执行命令: {cmd_str}")
    print("=" * 60)
    
    # 使用 subprocess 执行
    result = subprocess.run(cmd_str, shell=True, env=env, text=True)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"❌ 训练失败，退出码: {result.returncode}")
        print("=" * 60)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
