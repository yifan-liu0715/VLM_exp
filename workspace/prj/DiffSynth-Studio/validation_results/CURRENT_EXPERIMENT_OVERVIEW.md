# Wan2.2-TI2V-5B LoRA 当前实验总览

更新时间：2026-07-12（Asia/Shanghai）

## 1. 当前结论

快速训练得到的 LoRA **确实改变了模型行为**：参考图首帧一致性和时序稳定性略有提升，目标运动幅度误差下降；但强融合的 epoch-1 checkpoint 同时明显压制运动并降低锐度。因此，现有 quick LoRA 只能证明训练链路有效，还不能证明质量全面优于原模型。

当前 Pareto 候选是：

1. `epoch-0, alpha=1.0`：锐度保留最好，稳定性与运动匹配提升明显。
2. `epoch-1, alpha=0.5`：运动保留更好，整体更保守。

基于这一结果，更低学习率、无源数据泄漏的正式 v2 训练和全部 8 个 checkpoint 验证均已完成。当前正在对最终 top-2 执行额外 seed 稳健性复核。

> 重要限制：下面的 quick 指标只有 3 个训练域样本，适合诊断，不是无偏泛化结论。正式结论必须以 v2 source-disjoint 验证集、盲评和更完整指标为准。

## 2. Quick LoRA 对比结果

所有变体均使用相同条件：`576×320`、`33` 帧、`seed=42`、`tiled=True`。表中 `↑` 表示越高越好，`↓` 表示越低越好。

| 变体 | Ref SSIM ↑ | 锐度 ↑ | 平均光流 | 活跃运动占比 ↑ | Warp error ↓ | 目标运动 MAE ↓ | 相对 base 的判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| Base | 0.9598 | 415.0 | 1.157 | 0.969 | 0.01384 | 0.761 | 基线 |
| Epoch-1, α=1.00 | **0.9614** | 368.8 | 0.857 | 0.635 | **0.01095** | **0.559** | 最稳定、运动匹配误差最低，但运动抑制最强 |
| Epoch-0, α=1.00 | 0.9606 | **390.0** | 0.896 | 0.750 | 0.01110 | 0.590 | 当前最均衡候选之一 |
| Epoch-1, α=0.50 | 0.9601 | 371.9 | 0.985 | **0.854** | 0.01261 | 0.671 | 更保留运动，改动较温和 |
| Epoch-1, α=0.75 | 0.9602 | 359.0 | **1.052** | 0.792 | 0.01246 | 0.742 | 光流接近 base，但锐度最低、目标运动收益很小 |

### 相对 Base 的可视化百分比

| 变体 | 首帧 SSIM 变化 | 锐度保留 | 平均光流保留 | 活跃运动保留 | Warp error 降低 | 目标运动 MAE 降低 |
|---|---:|---:|---:|---:|---:|---:|
| Epoch-1, α=1.00 | +0.00155 | 88.9% | 74.1% | 65.6% | **20.9%** | **26.6%** |
| Epoch-0, α=1.00 | +0.00082 | **94.0%** | 77.4% | 77.4% | 19.8% | 22.5% |
| Epoch-1, α=0.50 | +0.00023 | 89.6% | 85.2% | **88.2%** | 8.9% | 11.8% |
| Epoch-1, α=0.75 | +0.00039 | 86.5% | **91.0%** | 81.7% | 10.0% | 2.5% |

解读时不能把“光流越小”简单视为更好：光流下降可能代表抖动减少，也可能代表应有的动作被抑制。因此必须联合查看 `warp_error`、`target_motion_MAE`、清晰度和人工视频盲评。

## 3. 视觉对比索引

每个图从视频中抽取同一时刻的帧，便于观察人物一致性、运动变化、模糊和几何漂移。

### mymotion__002

![mymotion__002 comparison](mymotion__002_compare_02.png)

### mymotion__012

![mymotion__012 comparison](mymotion__012_compare_02.png)

Epoch/alpha 消融示例：

![mymotion__012 ablation](mymotion__012_ablation_frame_16.jpg)

### mymotion__015

![mymotion__015 comparison](mymotion__015_compare_02.png)

可交互总览：`lora-comparison-standalone.html`。

## 4. 正式训练状态

- 训练输出：`/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/output_models/wan2.2_lora_formal_v2_r16_lr5e5/`
- 数据划分：`data_splits/source_disjoint_v2/train.csv`
- 模型：Wan2.2-TI2V-5B，LoRA rank 16
- 学习率：`5e-5`
- 训练轮数：2
- 梯度累积：4
- 保存间隔：每 750 data steps
- 训练尺寸：`576×320`，33 帧
- GPU：物理 GPU 5
- 已完成 checkpoint：`step-750`、`1500`、`2250`、`3000`、`3750`、`4500`、`5250`、`5986`（每个约 80.7 MB）
- 自动验证队列：`validation_results/formal_val_v2/queue.log`
- 当前状态：训练完成；15/15 跨分辨率样本文件预检通过；135 个正式视频、8 组指标、bootstrap 排名、45 个盲评对及仪表盘均已完成

自动队列已对所有 `step-*.safetensors` 使用 15 个 source-disjoint 样本生成视频、计算指标，并建立 guardrail、Pareto 前沿和 checkpoint 排名。

### 正式 v2 最终排名（seed 42）

| Rank | Checkpoint | 锐度保留（95% CI） | 运动保留（95% CI） | Warp error 降低 | 目标运动 MAE 降低 | 当前判断 |
|---:|---|---:|---:|---:|---:|---|
| 1 | step-750 | 0.859 [0.741, 0.959] | 0.956 [0.722, 1.198] | 0.356 | 0.172 | 最保守、稳定性收益最高 |
| 2 | step-5250 | 0.882 [0.754, 0.990] | 0.893 [0.621, 1.162] | 0.308 | 0.204 | 锐度/目标运动匹配更均衡 |
| 3 | step-4500 | 0.875 [0.740, 0.993] | 0.827 [0.542, 1.107] | 0.339 | 0.216 | 目标运动误差改善最大，但运动压制更强 |
| 4 | step-5986 | 0.853 [0.736, 0.949] | 0.875 [0.584, 1.161] | 0.318 | **0.233** | 最终训练点，锐度接近下限 |

`step-750` 与 `step-5250` 已进入多 seed 和保持纵横比复核。所有 checkpoint 的 CI-robust guardrail 均未完全通过，主要原因是 15 个样本下锐度置信区间较宽，因此尚不能宣布单一 checkpoint 全面胜出。

子组审计显示：固定 `576×320` 时，纵向 dance 样本锐度保留显著低于 talking-face；而 talking-face 的运动活跃度下降较多。由于 LoRA 同时改善了目标运动 MAE，这既可能是去除过度运动，也可能是动作塌缩，必须由保持纵横比复核和盲评判定。

## 5. 当前文件结构

以下只列与本次训练和验证直接相关的文件，预训练权重分片未展开。

```text
/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/
├── scripts/
│   ├── validate_lora.py                 # 生成 base 与任意 LoRA 的确定性对比视频
│   ├── evaluate_lora.py                 # OpenCV 客观指标与逐样本/聚合统计
│   ├── create_grouped_splits.py         # 按源视频分组，生成无泄漏 train/val/test
│   ├── select_validation_subset.py      # 从 val 中选择 15 个分桶样本
│   ├── train_wan22_lora_formal.sh       # 正式 v2 训练入口
│   ├── run_checkpoint_validation.sh     # 单 checkpoint 的生成+评估
│   ├── queue_formal_validation.sh       # 等待训练结束并依次验证全部 checkpoint
│   ├── rank_checkpoints.py              # guardrail、Pareto 与探索性排序
│   ├── prepare_blind_review.py           # 随机左右顺序，生成隐藏标签 A/B 盲评包
│   ├── build_formal_dashboard.py         # 排名、指标条形图与逐样本视频仪表盘
│   ├── analyze_formal_subgroups.py        # 分辨率桶及 talking-face/dance 子组审计
│   ├── aggregate_multiseed.py             # seed+sample 两层 bootstrap 汇总
│   ├── run_multiseed_validation.sh        # top-2 的额外 seed 复核
│   ├── run_aspect_validation.sh           # top-2 保持原始桶比例的复核
│   └── evaluate_dover.py                  # 官方 DOVER-Mobile 批量 learned quality
├── data/
│   └── dataset_video_train/
│       └── metadata_train.csv           # 原始训练元数据
├── data_splits/
│   ├── source_disjoint_v1/              # 已识别源分组缺陷，不再作为正式依据
│   └── source_disjoint_v2/
│       ├── train.csv                     # 2993 行，270 个源组
│       ├── val.csv                       # 468 行，3 个源组
│       ├── test.csv                      # 455 行，6 个源组
│       ├── split_manifest.json
│       ├── validation_subset_15.json
│       └── validation_subset_15.txt
├── output_models/
│   ├── wan2.2_lora_quick/
│   │   ├── epoch-0.safetensors
│   │   └── epoch-1.safetensors
│   ├── wan2.2_lora_formal_r16_lr5e5/
│   │   └── ABORTED_SOURCE_LEAKAGE_V1.txt
│   └── wan2.2_lora_formal_v2_r16_lr5e5/
│       ├── training_args.json
│       ├── training.log
│       └── step-{750..5986}.safetensors  # 共 8 个正式 checkpoint
└── validation_results/
    ├── *_base.mp4                        # 原始模型视频
    ├── *_lora*.mp4                       # checkpoint/alpha 对比视频
    ├── validation_manifest*.json         # 每次生成的参数与输出清单
    ├── frame_checks/                     # 抽帧对比图
    ├── metrics/
    │   ├── objective_metrics.csv
    │   ├── comparison_summary.csv
    │   └── aggregate_by_variant.csv
    ├── formal_val_v2/
    │   └── queue.log                     # 正式验证队列日志，结果待训练完成
    ├── validation_report.md
    ├── frontier_progress_report.md
    └── lora_evaluation_protocol.md
```

本地查看副本位于：`C:\Users\86173\Documents\project\validation_preview\`。

## 6. `validate_lora.py` 工作流程

```text
解析参数
  ↓
检查预训练权重、LoRA、metadata 和参考图目录
  ↓
按 sample_id 从 metadata_train.csv 读取 prompt 与 reference_image，并按 metadata 中的真实桶路径解析参考图
  ↓
一次性加载 T5 + DiT + VAE + tokenizer（bfloat16 / CUDA）
  ↓
逐样本生成 base 视频
  ↓
将指定 LoRA 以 alpha 融合到同一 DiT
  ↓
用完全相同的 prompt/reference/seed/尺寸/帧数生成 LoRA 视频
  ↓
写出 MP4 与 validation_manifest*.json
```

固定生成参数：

- `height=320`
- `width=576`
- `num_frames=33`
- `seed=42`
- `tiled=True`
- `fps=15`
- `torch_dtype=torch.bfloat16`

主要 CLI 参数：

- `--sample-ids ID [ID ...]`：指定样本。
- `--num-samples N`：快速只跑前 N 个。
- `--lora-path ABS_PATH`：指定 checkpoint。
- `--lora-alpha FLOAT`：LoRA 融合强度。
- `--lora-label LABEL`：控制输出文件后缀。
- `--output-dir ABS_PATH`：指定验证目录。
- `--overwrite`：覆盖已有视频；默认跳过已有文件。
- `--preflight-only`：只检查 checkpoint、metadata 和全部样本文件，不加载模型。
- `--seed INT`：覆盖默认 seed 42，用于多 seed 稳健性复核。
- `--resolution-mode fixed|bucket`：固定 `576×320`，或按桶使用 `640×256 / 576×320 / 320×576`。

示例：

```bash
cd /nvmedata/workspace2/users/nina/prj/DiffSynth-Studio
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate v2v_train
export PYTHONPATH=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio:$PYTHONPATH

CUDA_VISIBLE_DEVICES=5 python3 \
  /nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/scripts/validate_lora.py \
  --sample-ids mymotion__002 mymotion__012 mymotion__015 \
  --lora-path /nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/output_models/wan2.2_lora_quick/epoch-1.safetensors \
  --lora-alpha 0.5 \
  --lora-label lora_epoch1_a0p50 \
  --output-dir /nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/validation_results
```

输出命名：

- `{sample_id}_base.mp4`
- `{sample_id}_{lora_label}.mp4`
- `validation_manifest_{lora_label}.json`

## 7. `evaluate_lora.py` 评估范围

当前客观指标包括：

- 参考图与生成首帧的 SSIM；
- Laplacian 清晰度均值/波动；
- Farneback 光流均值与 P95；
- 活跃运动帧占比；
- 前后向光流 warp error；
- 帧间绝对差；
- 与目标视频运动曲线的相关性、MAE 和幅度比；
- LoRA 与 base 的逐帧配对 SSIM 和各指标 delta。

该脚本会输出逐视频 CSV、逐样本对比 CSV、按变体聚合 CSV 和完整 JSON。它不把单一代理指标当成最终质量结论。

## 8. 下一阶段判定门槛

正式 checkpoint 排名先应用以下 guardrail：

- 锐度保留率 ≥ 0.85；
- 运动保留率 ≥ 0.70；
- 首帧参考 SSIM 相对 base 不低于 -0.005。

每项指标还会对 15 个 held-out 样本执行 10,000 次配对非参数 bootstrap，报告 95% CI 和超过 base 的概率。除了均值 guardrail，还会标记 CI 下界仍满足阈值的 `95% CI robust` checkpoint。

通过 guardrail 后再看 Pareto 前沿，并以随机化、隐藏标签的人工盲评作为最终选择依据。正式验证队列会自动为排名靠前的最多 3 个变体生成 `blind_review/`：其中 `pairs/` 和 `blind_review_sheet.csv` 可交给评审者，`blind_review_key.csv` 必须单独保管，避免泄露左右模型身份。`formal_validation_dashboard.html` 会集中展示排名及逐样本视频。下一步补充 DOVER-Mobile 技术/审美 learned quality，随后再评估 VBench-I2V / VisionReward 的部署成本，并在独立 test split 上复核最终 checkpoint。

此外已排队两层稳健性实验：

1. 用 seed 42 的全 checkpoint 结果选 top-2，再用 seed 123 和 20260712 复核，并联合当前 seed 42 做 seed+sample 层级 bootstrap。
2. 多 seed 完成后，对 top-2 使用保持原始桶纵横比的三种等成本分辨率复核，判断 portrait/dance 锐度下降究竟来自 LoRA 还是固定横屏几何失配。

DOVER-Mobile 使用项目私有 `.venvs/dover`，不会修改 `v2v_train`。它只作为技术质量、审美质量和融合质量的次级 learned signal，不能替代盲评。
