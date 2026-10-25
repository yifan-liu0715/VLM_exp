# SCAIL-2 Base / DPO LoRA 与 Scheme4 对照实验设计

## 目标

在远端服务器建立完全隔离、可复现的 SCAIL-2 推理环境，使用与 Scheme4 相同的三个测试样本，比较：

1. 现有 Scheme4；
2. 官方 SCAIL-2 Base；
3. 官方 SCAIL-2 Base + Bias-Aware DPO LoRA。

本实验不重新训练 SCAIL-2。官方当前只公开推理代码和已训练权重，训练代码尚未公开。

## 隔离目录

- 实验根目录：`/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803`
- Conda 环境：`/nvmedata/workspace2/users/nina/conda_envs/scail2_0803`
- 代码：`/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/code/SCAIL-2`
- 官方权重：`/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/weights/SCAIL-2`
- 预处理条件：`/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/inputs`
- 推理结果：`/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/validation`
- 日志与清单：`/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/logs`、`manifests`

环境、代码、权重、输入、日志和结果均不得写入原 DiffSynth-Studio 实验目录。原 Scheme4 文件只读。

## 官方实现与权重

- 使用官方 `sat-scail2` 分支。
- 使用官方 SCAIL-2 SAT checkpoint，不额外转换为另一份完整 safetensors，避免约 65GB 重复占用。
- 使用官方 `model/bias-aware-dpo-lora.pt`。
- 下载后记录仓库 commit、文件大小和 SHA256。
- DPO 权重只能叠加到同一 SCAIL-2 Base，不作为独立模型运行。

## 固定测试样本

沿用 Scheme4 native supplement 的三个 test 样本：

1. `mymotion__051`（1280×512 横屏）；
2. `mymotion__single_talking_face__datatrain__002`（1280×704 横屏）；
3. `mymotion__humandance__batch1_mingxing_cropped__002`（704×1280 竖屏）。

对每个样本读取 Scheme4 `run_manifest.json` 中已经核验的：

- source/driving video；
- target video；
- reference image；
- prompt；
- width、height、fps；
- Scheme4 输出路径与 adapter 哈希。

不得重新猜测 input/output 方向。

## SCAIL-2 条件准备

每个样本生成并保存：

- `ref.jpg`：原 Reference，保持比例，不拉伸；
- `ref_mask.jpg`：Reference 人物前景/绑定 Mask；
- `driving.mp4`：与 Scheme4 source 相同的视频；
- `rendered_v2.mp4`：端到端模式下为 driving video 的同内容副本；
- `rendered_mask_v2.mp4`：与 driving video 每帧对齐的人物 Mask；
- `prompt.txt`：描述生成后视频的长英文 Prompt。

Mask 生成需使用官方 SCAIL-Pose 流程。若自动分割失败，保存中间可视化并停止该样本，不以全白/全黑 Mask 静默继续。

## 公平推理设置

Base 与 Base+DPO 必须完全一致：

- 同一 reference、reference mask、driving video、driving mask、prompt；
- 同一随机种子，固定为 42；
- 同一输出 width/height；宽高均需被 32 整除；
- 同一 denoising steps、flow shift、CFG、solver；
- 同一帧范围与 fps；
- 唯一变量为是否加载官方 DPO LoRA。

优先沿用原始三个分辨率。帧数采用官方 SAT 配置实际支持且三个方法共同可对齐的最大前缀；如不能直接支持 97 帧，统一裁成官方支持长度，并在 manifest 中记录，不允许只裁 SCAIL-2 而不裁对照视频。

首次仅对 `mymotion__humandance__batch1_mingxing_cropped__002` 做低成本 smoke test。确认 Mask、模型加载、Base 和 DPO 均正常后，再运行三个完整样本。

## GPU 与资源约束

- 绝不访问物理 GPU0。
- 启动前重新查询 GPU；只使用空闲的非 GPU0 卡。
- 单个推理任务默认单卡；只有官方 SAT 入口明确要求多卡时才使用多张空闲非 GPU0 卡。
- 不抢占已有计算进程，不以显存占用低但 GPU 利用率高的卡作为空闲卡。
- `/nvmedata` 可用空间低于 30GB 时停止新增下载/写入，保留日志和已完成结果；不得删除共享文件。
- 模型、环境和完整结果预计不得超过约 120GB。

## 输出结构

每个样本保存：

- `Source.mp4`
- `Target.mp4`
- `Scheme4.mp4`
- `SCAIL2_Base.mp4`
- `SCAIL2_Base_DPO.mp4`
- `comparison_6grid.mp4`
- `run_manifest.json`
- `ffprobe.json`
- `frames/frame_000.png`、中间帧、末帧

六宫格必须带固定标签，避免将 Target、Base、DPO 或 Scheme4 混淆。

## 验证与判定

先做结构性验证：

- 视频可解码；
- 帧数、分辨率、fps 与 manifest 一致；
- 无白屏、全黑、非有限 latent、OOM 或 traceback；
- Base 和 DPO 使用相同初始噪声及采样配置；
- 输出和权重记录 SHA256。

再做多帧人工比较，重点观察：

- 人物身份与脸部相似度；
- 表情、嘴唇和眼睛同步；
- 手指与身体边缘；
- 服装、发型和局部纹理；
- 动作、遮挡、背景保持；
- 帧间闪烁及局部伪影。

结论分别回答：

1. SCAIL-2 Base 相比 Scheme4 的整体能力差距；
2. DPO LoRA 是否在同一 Base 上改善脸、眼、唇、手及伪影；
3. Scheme4 的主要瓶颈更接近数据/条件接口，还是缺少细节后训练。

本轮不计算或宣称 SSIM、LPIPS、身份相似度等数值指标，除非后续单独建立一致且有效的评估协议。

## 失败处理

- GitHub/Hugging Face 网络失败：保留下载日志，使用断点续传，不生成不完整权重的成功标记。
- 依赖冲突：只修复隔离 Conda 环境，不修改 `v2v_train`。
- OOM：先保留完整错误与参数；不得改用 GPU0。优先使用官方 offload/并行配置，不静默修改分辨率和帧数。
- DPO 格式不匹配：使用官方 SAT 分支直接加载 `.pt`；不使用来源不明的第三方转换权重。
- Mask 失败：保存 Mask 可视化并停止该样本，修复后重跑。

## 交付物

- 隔离环境导出文件；
- 官方代码 commit；
- 权重清单与 SHA256；
- 输入条件、Mask 和 Prompt；
- Base、Base+DPO、Scheme4 三组结果；
- 六宫格视频和抽帧；
- 每个样本的运行清单、ffprobe 报告和日志；
- 一份可直接发给导师的简短结论；
- 完成后复制紧凑结果到 `D:\Code\Wan2.2_vs_SCAIL2_<timestamp>`，不复制 81GB 官方权重和 Conda 环境。
