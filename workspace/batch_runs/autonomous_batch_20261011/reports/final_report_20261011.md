# autonomous_batch_20261011 最终批次报告

生成时间：2026-10-11T20:05:00+08:00

## 执行结论

远端连接、原片传输、机械候选生成和候选文件审计已完成。所有结果仍停留在人工审核前的候选阶段；未启动关键点检测和训练。

## 输入与首轮覆盖

- 原片：130 个，约 36.65 GiB，约 2.7326 小时。
- 首轮机械扫描：130/130 个源片完成。
- 发现候选文件的源片：129 个；未产生候选文件的源片：1 个（`DJI_20261004205501_0126_D`，未检测到手部，保留为审阅记录）。
- 以源目录、每源 manifest 和 SHA256 收据判定的未处理源片：0 个。
- 源片失败：0 个；detector_failed：0 个。

## 候选与状态

- 候选记录：597 条。
- 实际候选视频：596 条。
- `motion_proposal`：597 条。
- `second_pass_required`：597 条。
- `semantic_valid_short`：0 条。
- `video_approved_pending_loader`：0 条。
- `review_needed`：0 条；当前候选尚未进入该晋级状态，均由 `motion_proposal/second_pass_required` 承载。
- `reject`：0 条。
- `accepted_for_training`：0 条。

## 格式和隐私

- 596/596 个实际候选通过 1280×720、CFR 16 FPS、H.264、无音频检查。
- 596/596 个实际候选完整解码通过。
- 缺失候选文件 0，临时 `.encoding.mp4` 0，manifest 重复路径 0。
- 人脸检测命中 518 条；`privacy_reviewed=true` 为 0。
- 尚未执行固定去脸裁剪和人工主体清晰度确认，因此任何候选都不能视为隐私通过。

## 方法和监控

- 使用了低分辨率 3 FPS/640×360 初筛定位；低分辨率结果没有作为语义通过依据。
- 未使用 VLM。
- 远端恢复会话 `batch_recovery_20261011_ui` 已正常退出；整批候选审计会话也已正常退出。
- 最终检查时 GPU 2/3 空闲，其他 GPU 有其他任务；远端 `/nvmedata` 仅约 102G 可用空间（99% 已用），因此没有再启动新的 worker。

## 完整路径

- 本地候选目录：`G:\data\clips_new_10_04\autonomous_batch_20261011\clips`
- 本地源 manifest：`G:\data\clips_new_10_04\autonomous_batch_20261011\source_manifest.jsonl`
- 本地候选汇总 manifest：`G:\data\clips_new_10_04\autonomous_batch_20261011\clips_manifest_all.jsonl`（597 条）
- 本地传输 manifest：`G:\data\clips_new_10_04\autonomous_batch_20261011\transfer_manifest.jsonl`
- 本地计划/日志/审核目录：`G:\data\clips_new_10_04\autonomous_batch_20261011\plans`、`G:\data\clips_new_10_04\autonomous_batch_20261011\logs`、`G:\data\clips_new_10_04\autonomous_batch_20261011\reviews`、`G:\data\clips_new_10_04\autonomous_batch_20261011\uncertain`
- 本地候选审计：`G:\data\clips_new_10_04\autonomous_batch_20261011\reports\full_candidate_audit_20261011.md`
- 本地最终 JSON：`G:\data\clips_new_10_04\autonomous_batch_20261011\reports\final_report_20261011.json`
- 本地二审队列：`G:\data\clips_new_10_04\autonomous_batch_20261011\reports\recovery_v4_second_pass_queue_20261011.jsonl`
- 远端候选目录：`/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/clips`
- 远端报告目录：`/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/reports`

## 下一道闸门

人工审核候选的视频因果链、动作边界、多动物归属和人脸裁剪；只有审核通过的片段才进入关键点检测。

## 本轮人工反馈

- 0132 的第 001 条已记录为 reject：没有第一人称手部—动物互动。
- 0132 的第 004 条已记录为 recut_required：揉脸、摸头、戳鼻子等动作被合并。
- 反馈记录：`G:\data\clips_new_10_04\autonomous_batch_20261011\reviews\feedback_0132_20261011.jsonl`
- 流程缺口说明：`G:\data\clips_new_10_04\autonomous_batch_20261011\reports\skill_gap_0132_20261011.md`
