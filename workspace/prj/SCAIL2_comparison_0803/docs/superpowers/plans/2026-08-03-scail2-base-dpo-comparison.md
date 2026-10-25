# SCAIL-2 Base / DPO LoRA Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task with review checkpoints.

**Goal:** 在远端服务器建立完全隔离、可复现的 SCAIL-2 官方推理环境，用同一组三个 V2V 样本比较现有 Scheme4、官方 SCAIL-2 Base、官方 SCAIL-2 Base + Bias-Aware DPO LoRA，并产出可人工检查的对比视频和完整运行证据。

**Architecture:** 保留现有 Scheme4 目录只读；在新的实验根目录中分别保存官方代码、隔离 Conda 环境、官方权重、固定输入、运行日志、manifest 和验证视频。SCAIL-2 Base 与 DPO 组使用同一官方推理链路，唯一变量是 DPO LoRA；Scheme4 直接复用已完成的原生分辨率结果。先对人舞样本做门控 smoke test，通过后再运行三分辨率样本。

**Tech Stack:** Linux/SSH、Conda、PyTorch、SAT/SwissArmyTransformer、SCAIL-2、Wan2.1-I2V-14B、SCAIL-Pose、FFmpeg/ffprobe、Git、SHA256。

## Global Constraints

- 远端主机：`student@115.190.111.129`。
- 实验根目录：`/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803`。
- Conda 前缀：`/nvmedata/workspace2/users/nina/conda_envs/scail2_0803`。
- 不使用物理 GPU0；每次启动前重新检查 GPU，且只绑定空闲的非 GPU0。
- `/nvmedata` 可用空间低于 30 GB 时停止新增写入，保全日志，不删除共享文件。
- 不修改、不覆盖 Scheme4 原目录与权重。
- 本轮不重新训练 SCAIL-2，也不声称 SSIM、LPIPS 或人脸身份指标获得提升。
- Base 与 DPO 的提示词、输入、mask、随机种子、帧数、采样器和输出尺寸必须完全一致；唯一变量是 DPO LoRA。
- 若官方 SAT 分支无法证明 DPO 权重确实加载，则停止 DPO 推理并报告，不用“文件存在”冒充“权重生效”。

---

### Task 1: 建立目录、资源门控与不可变清单

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/resource_audit.txt`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/source_assets.sha256`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/logs/`

**Step 1: 创建隔离目录**

```bash
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
mkdir -p "$ROOT"/{code,weights,inputs,validation,logs,manifests,scripts,tests,docs/superpowers/plans}
```

Expected: 所有目录存在，且没有写入 DiffSynth-Studio 的 Scheme4 实验目录。

**Step 2: 记录磁盘、GPU 和软件状态**

```bash
{
  date -Is
  hostname
  df -h / /tmp /nvmedata /mnt/highspeed
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
  conda --version
  git --version
  ffmpeg -version | head -n 1
} > "$ROOT/manifests/resource_audit.txt"
```

Expected: `/nvmedata` 可用空间不少于 30 GB；GPU0 即使空闲也不进入候选 GPU。

**Step 3: 固定 Scheme4 资产哈希**

```bash
SCHEME4=/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_reference_independent_posenc_scheme4_0729
find "$SCHEME4/recommended/final_adapter" \
     "$SCHEME4/validation/native_supplement_scheme3_conditions_0730" \
     -type f \( -name '*.safetensors' -o -name '*.mp4' -o -name 'run_manifest.json' \) \
     -print0 | sort -z | xargs -0 sha256sum > "$ROOT/manifests/source_assets.sha256"
```

Expected: 清单包含 Scheme4 final adapter、三组视频和各自 manifest；原文件时间戳不变。

**Step 4: 提交门控证据**

```bash
cd "$ROOT"
git add manifests
git commit -m "chore: record SCAIL2 comparison resource gate"
```

---

### Task 2: 获取并固定官方 SCAIL-2 代码

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/code/SCAIL-2/`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/scail2_source.txt`

**Step 1: 克隆官方 SAT 分支并固定 commit**

```bash
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
git clone --filter=blob:none --single-branch --branch sat-scail2 \
  https://github.com/zai-org/SCAIL-2.git "$ROOT/code/SCAIL-2"
git -C "$ROOT/code/SCAIL-2" checkout 3dee8c4808057690393a189836c508b966ec282b
git -C "$ROOT/code/SCAIL-2" rev-parse HEAD | tee "$ROOT/manifests/scail2_source.txt"
```

Expected: HEAD 精确为 `3dee8c4808057690393a189836c508b966ec282b`。网络中断时使用同一命令重试，不切换到非官方镜像源码。

**Step 2: 审计官方入口和配置**

```bash
cd "$ROOT/code/SCAIL-2"
rg -n "bias-aware-dpo|lora|num_frames|sampling_image_size|num_steps|shift_scale|cfg" \
  README.md scripts configs sat sample_video.py > "$ROOT/manifests/official_entry_audit.txt"
```

Expected: 能定位官方采样入口、81 帧配置、50 denoise steps、Flow shift 5、CFG 4，以及 DPO 相关代码或权重说明。

---

### Task 3: 创建隔离 Conda 环境并做 CPU 导入测试

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/conda_explicit.txt`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/tests/test_imports.py`

**Step 1: 创建环境**

```bash
ENV=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803
conda create -y -p "$ENV" python=3.11 pip
conda run -p "$ENV" python -m pip install --upgrade pip setuptools wheel
conda run -p "$ENV" python -m pip install -r \
  /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/code/SCAIL-2/requirements.txt
```

Expected: 环境安装不修改 `v2v_train`。Python 3.11 是执行时根据官方
`scipy>=1.16.3` 解析结果确认的最低兼容版本；Python 3.10 只能解析到
SciPy 1.15.3，不能满足官方 requirements。

**Step 2: 写入最小导入测试**

```python
import torch
import torchvision
import diffusers
import transformers
import sat
import av
import decord

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
```

**Step 3: 运行测试并冻结环境**

```bash
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803
conda run -p "$ENV" python "$ROOT/tests/test_imports.py" | tee "$ROOT/logs/import_test.log"
conda list -p "$ENV" --explicit > "$ROOT/manifests/conda_explicit.txt"
```

Expected: 所有导入成功，`torch.cuda.is_available()` 为 `True`；失败则保留完整 pip/导入日志后停止。

---

### Task 4: 下载官方模型并验证完整性

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/weights/SCAIL-2/`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/model_files.sha256`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/model_download.json`

**Step 1: 下载前再次检查空间**

```bash
AVAIL_KB=$(df --output=avail /nvmedata | tail -n 1)
test "$AVAIL_KB" -ge $((110 * 1024 * 1024))
```

Expected: 下载前至少 110 GB 可用；不足则不下载并报告。

**Step 2: 使用官方 Hugging Face 仓库下载到隔离目录**

```bash
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803
conda run -p "$ENV" huggingface-cli download zai-org/SCAIL-2 \
  --local-dir "$ROOT/weights/SCAIL-2" --resume-download
```

Expected: 官方 Base checkpoint、`bias-aware-dpo-lora.pt`、T5、VAE、CLIP 等必需文件齐全。若复用共享 T5/VAE，必须先确认 SHA256 与官方文件相同，并在 manifest 写明符号链接来源。

**Step 3: 校验 DPO 文件和全量哈希**

```bash
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
test "$(sha256sum "$ROOT/weights/SCAIL-2/model/bias-aware-dpo-lora.pt" | awk '{print $1}')" = \
  cf29934c72fee6b24c3a5a82e8a9a4f3f4e93d76d988c9a20fac4e2b19e8fa09
find "$ROOT/weights/SCAIL-2" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/manifests/model_files.sha256"
```

Expected: 官方 DPO SHA256 精确匹配；任何关键文件不完整则不进入 GPU 阶段。

---

### Task 5: 从 Scheme4 manifest 固定三组输入并生成 SCAIL-Pose 条件

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/scripts/prepare_inputs.py`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/scripts/validate_masks.py`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/inputs/<sample_id>/`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/input_manifest.json`

**Step 1: 先写输入清单测试**

测试必须断言以下映射，不允许按文件名猜测方向：

```text
source/driving = dataset/.../output_video/<sample>.mp4
target         = dataset/.../input_video/<sample>.mp4
reference      = dataset/.../reference/<sample>.jpg
```

固定样本：

```text
mymotion__051                                      1280x512
mymotion__single_talking_face__datatrain__002     1280x704
mymotion__humandance__batch1_mingxing_cropped__002 704x1280
```

**Step 2: 实现 manifest 驱动的复制/链接**

`prepare_inputs.py` 读取 Scheme4 三个 `run_manifest.json`，为每个样本建立只读符号链接并保存 prompt、source、target、reference、原始分辨率、fps、帧数和 SHA256。不得重新定义标签方向。

**Step 3: 用官方 SCAIL-Pose 生成条件**

对每个样本生成：

```text
rendered_v2.mp4
rendered_mask_v2.mp4
ref_mask.png
```

生成命令必须记录到每个样本的 `pose_command.txt`，标准输出和错误写入 `logs/pose_<sample>.log`。

**Step 4: 验证 mask**

`validate_masks.py` 至少检查：文件可读、帧数与 driving video 对齐、mask 非全黑/全白、前景面积比例有限、0/中间/末帧存在人体区域，并导出三张 overlay PNG。

Expected: 三个样本全部通过才可继续；任一样本 mask 明显错位时停止，不运行生成模型。

---

### Task 6: 实现并证明 Base/DPO 加载路径

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/scripts/audit_dpo_checkpoint.py`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/scripts/run_scail2.py`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/tests/test_dpo_applied.py`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/dpo_load_report.json`

**Step 1: 审计官方 DPO checkpoint**

`audit_dpo_checkpoint.py` 使用 CPU 加载 `bias-aware-dpo-lora.pt`，记录顶层键、张量数量、shape/dtype、LoRA rank、非零比例和有限值检查，不把大张量打印进日志。

**Step 2: 实现单一推理入口**

`run_scail2.py` 封装官方 `sample_video.py` 和配置；参数至少包括：

```text
--sample-manifest
--output-dir
--seed 42
--mode base|dpo
--dpo-path
--num-frames
--gpu-id
```

Base 模式明确禁止加载 DPO；DPO 模式必须调用官方 SAT LoRA 注入路径。Base 与 DPO 共享相同模型配置、输入预处理和 sampler 参数。

**Step 3: 写入 DPO 生效测试**

`test_dpo_applied.py` 必须验证：

1. checkpoint 中所有预期 LoRA tensor 均被消费；
2. 无关键 missing/unexpected keys；
3. 注入后 LoRA 参数非零；
4. 在固定微型输入、固定 seed 下，Base 与 DPO 的一次前向输出差异大于数值噪声；
5. 保存 Base/DPO 可训练或注入参数数量及加载后的 SHA256 摘要。

Expected: 生成 `dpo_load_report.json` 且 `applied=true`。不满足时停止，不能生成带有 DPO 标签的视频。

---

### Task 7: 运行人舞样本门控 smoke test

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/validation/smoke_humandance/`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/logs/smoke_base.log`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/logs/smoke_dpo.log`

**Step 1: 选择 GPU**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

选择空闲非 GPU0，仅以 `CUDA_VISIBLE_DEVICES=<physical_id>` 启动；在运行 manifest 同时保存物理 ID 和进程可见逻辑 ID。

**Step 2: 运行 Base**

```bash
CUDA_VISIBLE_DEVICES=<physical_id> conda run -p "$ENV" python "$ROOT/scripts/run_scail2.py" \
  --sample-manifest "$ROOT/inputs/mymotion__humandance__batch1_mingxing_cropped__002/sample.json" \
  --output-dir "$ROOT/validation/smoke_humandance/base" \
  --mode base --seed 42 --num-frames 81 --gpu-id 0 \
  > "$ROOT/logs/smoke_base.log" 2>&1
```

**Step 3: 运行 DPO**

同一命令仅将 `--mode` 改为 `dpo` 并传入官方 DPO 路径。

**Step 4: 门控检查**

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,nb_frames \
  -of json <output.mp4>
```

抽取第 0/40/80 帧，检查：非白屏/黑屏、无 NaN、人物区域存在、mask 没有明显错位、Base 与 DPO 文件哈希不同、DPO load report 为 applied。

Expected: 两组均通过才运行完整三样本；失败时保全日志和产物并停止。

---

### Task 8: 运行三分辨率正式对比

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/validation/<sample_id>/SCAIL2_Base.mp4`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/validation/<sample_id>/SCAIL2_Base_DPO.mp4`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/validation/<sample_id>/Scheme4.mp4`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/validation/<sample_id>/run_manifest.json`

**Step 1: 固定帧长策略**

官方短序列以 81 帧运行。为公平比较，Scheme4、Source、Target 均裁取同一前 81 帧；若后续启用官方 97 帧分段模式，三组必须使用同一 97 帧区间，并在 manifest 记录 81 帧窗口、5 帧 overlap 和拼接策略。

**Step 2: 对三个样本逐一运行 Base 与 DPO**

推理参数固定为：seed 42、50 denoise steps、Flow shift 5、CFG 4、官方 SAT model-native resize/crop、相同 prompt、相同 pose/mask。每次重新记录 GPU；允许在两个空闲非 GPU0 上并行 Base/DPO，但日志和输出目录必须隔离。

**Step 3: 复用 Scheme4 结果**

从 Scheme4 native supplement 目录复制对应 `Scheme4_NativePEInterpolated.mp4` 到比较目录，并验证 SHA256 与 Task 1 清单一致；不重新推理 Scheme4。

Expected: 每个样本有 Source、Target、Scheme4、SCAIL2 Base、SCAIL2 DPO 及完整 manifest。

---

### Task 9: 生成六宫格、审计并复制紧凑结果到本地

**Files:**
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/scripts/build_comparison.py`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/scripts/audit_results.py`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/validation/<sample_id>/comparison_6grid.mp4`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/final_audit.json`
- Create: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/REPORT.md`

**Step 1: 生成统一标注的六宫格**

布局固定为：

```text
Source | Reference | Target
Scheme4 | SCAIL2 Base | SCAIL2 Base + DPO
```

所有格子保持各自长宽比并 padding，不拉伸；视频采用共同的帧数和 fps。

**Step 2: 自动审计**

`audit_results.py` 检查每个 MP4：可解码、帧数/fps/尺寸、非空、像素均值和方差有限、不是白屏/黑屏；抽取 0/中间/末帧；计算 SHA256。扫描全部日志中的 `Traceback|CUDA out of memory|NCCL|No space|nan|inf`。

**Step 3: 人工检查并形成限定结论**

`REPORT.md` 按三样本记录：人物身份/脸部、发型衣服、动作保持、背景、时序闪烁和伪影。只写可由视频直接支持的定性结论；不把 Flow Matching loss 当成 SSIM 或感知质量指标。

**Step 4: 复制紧凑结果到本地**

```powershell
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$dst = "D:\Code\Wan2.2_vs_SCAIL2_$stamp"
New-Item -ItemType Directory -Force $dst | Out-Null
scp -r student@115.190.111.129:/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/validation $dst
scp -r student@115.190.111.129:/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests $dst
scp -r student@115.190.111.129:/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/scripts $dst
scp student@115.190.111.129:/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/REPORT.md $dst
```

不复制 81 GB 官方模型、Conda 环境或大型缓存。

**Step 5: 校验远端/本地关键哈希并提交可复现代码**

远端与本地分别计算 final manifest、DPO load report、三组六宫格和脚本 SHA256，要求完全一致。随后在实验仓库提交脚本、测试、配置、manifest、报告和小型预览，不提交模型权重或大视频。

```bash
cd /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
git add scripts tests manifests docs REPORT.md
git commit -m "feat: add reproducible SCAIL2 base and DPO comparison"
```

Expected: `final_audit.json` 总状态为 passed，本地备份可打开，且报告清楚区分 Scheme4、SCAIL-2 Base 和 DPO 的观察结果。
