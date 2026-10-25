# Wan2.2-TI2V-5B LoRA 验证报告

验证日期：2026-07-12  
项目根目录：`/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio`

## 1. 验证目标

使用完全一致的输入、提示词与生成参数，对比 Wan2.2-TI2V-5B 基座模型和 `epoch-1.safetensors` LoRA 模型的视频生成结果，判断 LoRA 是否被正确加载、是否改变生成轨迹，以及是否改善人物访谈场景的运动与结构稳定性。

## 2. 当前相关文件结构

```text
/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/
├── data/
│   └── dataset_video_train/
│       └── metadata_train.csv
├── output_models/
│   └── wan2.2_lora_quick/
│       ├── training_args.json
│       ├── epoch-0.safetensors
│       └── epoch-1.safetensors            # 本次验证使用
├── pretrain_weights/
│   └── Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B/
│       ├── models_t5_umt5-xxl-enc-bf16.pth
│       ├── diffusion_pytorch_model-00001-of-00003.safetensors
│       ├── diffusion_pytorch_model-00002-of-00003.safetensors
│       ├── diffusion_pytorch_model-00003-of-00003.safetensors
│       ├── Wan2.2_VAE.pth
│       └── google/umt5-xxl/               # tokenizer
├── scripts/
│   └── validate_lora.py                   # 完整验证脚本
└── validation_results/
    ├── validation_manifest.json
    ├── mymotion__002_base.mp4
    ├── mymotion__002_lora.mp4
    ├── mymotion__012_base.mp4
    ├── mymotion__012_lora.mp4
    ├── mymotion__015_base.mp4
    ├── mymotion__015_lora.mp4
    └── frame_checks/
        ├── mymotion__002_compare_01.png ... compare_03.png
        ├── mymotion__012_compare_01.png ... compare_03.png
        └── mymotion__015_compare_01.png ... compare_03.png
```

测试参考图从以下绝对目录读取：

```text
/nvmedata/workspace2/users/nina/dataset_video_train/1280x704/reference/
```

## 3. 验证参数

| 参数 | 值 |
|---|---:|
| height | 320 |
| width | 576 |
| num_frames | 33 |
| seed | 42 |
| tiled | True |
| fps | 15 |
| 推理步数 | pipeline 默认 50 步 |
| LoRA alpha | 1 |
| GPU | 物理 GPU 5 |

## 4. 验证脚本工作流程

`scripts/validate_lora.py` 的执行逻辑如下：

1. 检查所有预训练权重、tokenizer、LoRA、元数据和参考图绝对路径。
2. 从 `metadata_train.csv` 读取指定样本，使用 `video` 文件名提取 `sample_id`。
3. 将元数据中的参考图文件名映射到指定的测试参考图目录。
4. 只加载一套 Wan2.2-TI2V-5B pipeline，避免两套 5B 模型同时驻留显存。
5. 对全部样本先生成 `{sample_id}_base.mp4`。
6. 调用 `pipe.load_lora(pipe.dit, epoch-1.safetensors, alpha=1)`，共融合 300 个 LoRA 张量。
7. 使用相同 prompt、参考图、seed 和生成参数生成 `{sample_id}_lora.mp4`。
8. 写出 `validation_manifest.json`，记录参数与全部输出路径。

默认样本：

```text
mymotion__002
mymotion__012
mymotion__015
```

可用参数：

```text
--sample-ids ID [ID ...]   指定样本
--num-samples N            只运行前 N 个样本
--overwrite                覆盖已存在的视频
```

复现实验命令：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate v2v_train
cd /nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
export PYTHONPATH=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false
python scripts/validate_lora.py --num-samples 3
```

## 5. 输出核验

六个视频均为 H.264、576×320、15 fps、33 帧、时长 2.2 秒。

| 样本 | base 大小 | LoRA 大小 | Base–LoRA SSIM |
|---|---:|---:|---:|
| mymotion__002 | 135,902 B | 109,088 B | 0.749830 |
| mymotion__012 | 112,556 B | 77,829 B | 0.724703 |
| mymotion__015 | 132,672 B | 96,140 B | 0.698809 |

SSIM 仅表示两段视频的像素相似度；数值低于 1 证明 LoRA 改变了生成轨迹，但不能单独代表质量更高。

## 6. 视觉对比结果

每张抽帧图均为左侧 base、右侧 LoRA，并选择第 1、17、33 帧。

- `mymotion__002`：LoRA 中后段背景与人物轮廓更稳定；base 末帧出现更明显的背景撕裂和手臂伪影。
- `mymotion__012`：两者都生成了右侧人物讲话/抬手动作；LoRA 末帧背景更干净，base 出现局部蓝色伪影和结构漂移。
- `mymotion__015`：LoRA 对“右侧人物讲话并做手势”的语义响应更直接；两者都有快速手部运动模糊，但 LoRA 的身体结构和背景连续性更好。

LoRA 文件通常更小，可能来自画面更平滑、伪影更少，也可能意味着高频细节或运动幅度降低，因此不能仅依据文件大小判断优劣。

## 7. 当前结论

LoRA 已正确加载并明显改变输出，三个样本均表现出较好的中后段结构稳定性，当前快速训练具有正向效果，值得继续正式训练。

但这些样本来自训练元数据，只能证明训练域内效果。正式训练前应增加独立验证集，至少使用 5–10 个未参与训练的参考图，并同时检查人物身份一致性、手部/面部畸变、背景稳定性、动作与 prompt 的一致性，以及较长视频下的时间连续性。
