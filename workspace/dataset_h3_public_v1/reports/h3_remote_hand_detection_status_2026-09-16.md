# H3 远程手部关键点当前状态（2026-09-16）

## 已完成

- 公开 EPIC-KITCHENS pilot manifest：100 条动作样本，70 个唯一源视频。
- 源视频下载：70/70 个文件完整，`.aria2` 部分文件数为 0。
- MediaPipe Hand Landmarker：100/100 条 JSON 已生成。
- 总采样帧数：17,129；检测到的手实例：19,339。
- 每个检测到的手均为 21 个点，每个点为归一化 `(x, y, z)`。
- `summary.json` 与 manifest 均为 100 条，sample_id 已完成一致性核对。
- JSON 解析、非空帧、21 点/3 坐标结构校验全部通过。

## 当前协议

- 点顺序固定为 MediaPipe HandLandmark `0..20`。
- 保留每只手的 `handedness`、`handedness_score`、21 点坐标和逐帧时间戳。
- 没有检测到手时保留空列表，不填充伪关键点。
- 左右手身份暂不强行修正，后续需要跨帧轨迹匹配。

## 已知数据问题

- `epic_P30_113_117` 使用公开动作区间短片，已标记 `source_access=hf_action_clip`、`window_complete=false`；关键点有效，但不含 manifest 额外的上下文帧。
- 其余 99 条使用完整源视频窗口。
- 全部帧中有少量双手原始 handedness 相同的情况，不能直接把该标签当作稳定的左右槽位。

## 训练状态

- H3 Stage 1/Stage 2 模型训练尚未启动，按照当前要求先完成手势检测点确认。
- 当前没有手部检测进程、片段补检进程或 H3 训练进程运行。
- 关键点结果目录：
  `/nvmedata/workspace2/users/nina/dataset_h3_public_v1/derived/epic_pilot_100/hand_keypoints/`

## 下一步

先对关键点做可视化抽查和跨帧左右手轨迹匹配，生成固定的 `left/right` 槽位与有效性 mask；完成人工确认后，再讨论是否进入 H3 训练。
