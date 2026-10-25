# H3 手部关键点协议

当前公开 EPIC-KITCHENS pilot 的手部检测结果统一采用 MediaPipe Hand Landmarker 的 21 点拓扑。每一帧的每只手都必须有 `points[0..20]`，每个点为归一化坐标 `[x, y, z]`。

## 点顺序

| index | name | 中文含义 |
| ---: | --- | --- |
| 0 | WRIST | 手腕 |
| 1 | THUMB_CMC | 拇指腕掌关节 |
| 2 | THUMB_MCP | 拇指掌指关节 |
| 3 | THUMB_IP | 拇指指间关节 |
| 4 | THUMB_TIP | 拇指指尖 |
| 5 | INDEX_FINGER_MCP | 食指掌指关节 |
| 6 | INDEX_FINGER_PIP | 食指近端指间关节 |
| 7 | INDEX_FINGER_DIP | 食指远端指间关节 |
| 8 | INDEX_FINGER_TIP | 食指指尖 |
| 9 | MIDDLE_FINGER_MCP | 中指掌指关节 |
| 10 | MIDDLE_FINGER_PIP | 中指近端指间关节 |
| 11 | MIDDLE_FINGER_DIP | 中指远端指间关节 |
| 12 | MIDDLE_FINGER_TIP | 中指指尖 |
| 13 | RING_FINGER_MCP | 无名指掌指关节 |
| 14 | RING_FINGER_PIP | 无名指近端指间关节 |
| 15 | RING_FINGER_DIP | 无名指远端指间关节 |
| 16 | RING_FINGER_TIP | 无名指指尖 |
| 17 | PINKY_MCP | 小指掌指关节 |
| 18 | PINKY_PIP | 小指近端指间关节 |
| 19 | PINKY_DIP | 小指远端指间关节 |
| 20 | PINKY_TIP | 小指指尖 |

该顺序与 [MediaPipe 官方 HandLandmark 枚举](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/hand_landmarker/HandLandmark)一致，后续代码不得重新排序。

## 坐标和字段

- `x`、`y` 是图像宽高归一化坐标，范围通常为 `[0, 1]`，原点在左上角。
- `z` 是 MediaPipe 的相对深度坐标，不是世界坐标或摄像机绝对深度。
- `handedness` 和 `handedness_score` 保留检测器原始输出。
- `hand_count` 表示该帧检测到的手数；没有检测到手时 `hands=[]`，不填充伪关键点。
- 当前采样频率为 24 FPS；`time_sec` 使用源视频时间轴。

## 左右手处理规则

1. 原始 JSON 不修改检测器的 Left/Right 判断。
2. 一帧内两只手出现相同标签时，记录为手身份冲突；不能据此把两只手都当作同一只手。
3. 后续需要固定形状张量时，先做跨帧手轨迹匹配，再生成 `left/right` 两个槽位；匹配失败的槽位保留为空，并同步提供有效性 mask。
4. 关键点坐标和手身份是两个独立质量字段，不能用左右标签修正点编号。

## 当前结果审计

- 100 条样本、17,129 帧、19,339 个手实例。
- 所有检测到的手均为 21 点、每点 3 个坐标值。
- 共有 6,402 帧同时检测到两只手，其中 632 帧的原始左右标签相同；这些帧需要后续轨迹匹配后才能形成可靠的左右槽位。
