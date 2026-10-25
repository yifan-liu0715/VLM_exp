# H3 服务器训练准备状态（2026-09-13）

## 已完成

- 服务器：`student@115.190.111.129`，主机 `is-dcrc324uvsgyc6mv-devmachine-0`。
- 数据目录：`/nvmedata/workspace2/users/nina/dataset_h3_public_v1/`。
- 训练准备目录：`/nvmedata/workspace2/users/nina/h3_training_v1/`。
- 已建立 `code/`、`configs/`、`runs/`、`logs/`，并上传 Stage 1/Stage 2 训练代码审计副本。
- 已上传采样矩阵：`configs/h3_sampling_matrix_2026-09-13.md`。
- EPIC 预览视频已放在数据目录的 `derived/epic_review_pilot/`，用于人工格式验收。

## 当前不能直接启动训练的原因

1. 8 张 A100 当前全部被其他任务占用；不能抢占 GPU 或杀掉其他进程。
2. 远端未找到 LiveAnimate 训练所需的 Wan2.2-Animate-14B 权重；现有 DiffSynth-Studio 目录只有 Wan2.2-TI2V-5B，不能直接替代。
3. H3 数据目前还没有完成手部关键点、骨架控制视频、响应标签和训练 manifest，不能把 EPIC 元数据候选直接喂给 Stage 1/Stage 2。
4. `/nvmedata` 可用空间约 278 GB。先按 manifest 验收和抽取训练窗口，避免复制全量原视频。

另外，现有 LiveAnimate 数据接口按 whole-body `pose` + `face` + `motion` 三路输入设计，而 H3 计划主要是第一人称双手控制。即使把手部骨架绘制成 pose mp4 在张量形状上接通，也必须先做 teacher 可视化 smoke test，确认预训练 pose adapter 能理解这种控制语义；第一人称没有脸时还需要生成兼容的 face 输入或修改 reader 允许 `face=None`。

## 启动顺序

1. 确认 1 张以上非零 GPU 空闲，先做 20–50 条样本 smoke test。
2. 将通过验收的片段统一到 1080p/24fps，生成 `rgb_firstperson`、`hand_keypoints`、`control_video` 和 `response_label`。
3. 先跑 Stage 1 控制条件 LoRA/SFT，确认输入字段和 loss 正常；再用 Stage 1 checkpoint 跑 Stage 2 causal/DMD 蒸馏。
4. 记录训练样本数、有效率、显存、吞吐和生成延迟，再决定是否扩大到约 1100 条自制 + 4000–5000 条公开片段。

训练命令不能在模型权重和数据 manifest 未就绪前启动；当前准备工作只创建目录和上传代码，不代表训练已经运行。
