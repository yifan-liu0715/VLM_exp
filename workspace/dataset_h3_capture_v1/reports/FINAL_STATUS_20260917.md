# 282-video capture processing status

Updated: 2026-09-17

## Completed

- Source: `G:\论文数据集\raw_data\DCIM\DJI_001`.
- Source inventory: 282/282 videos passed `ffprobe`; original MP4 files were not modified or moved.
- Base motion candidates: 1,882 clips from 264 source videos.
- Supplement coverage: 18 clips for the 18 sources missed by the first selector: 7 `coverage_only` clips (4--5 seconds) and 11 `short_source` clips (the complete source is shorter than 4 seconds).
- Total remote clips: 1,900; total source coverage: 282/282.
- Base export validation: 1,882/1,882 passed. Supplement export validation: 18/18 passed.
- Export contract: H.264 MP4, 24 FPS, no audio; landscape 1280x720 and portrait 720x1280.
- Remote clip SHA256: 1,900/1,900 matched the full manifest.
- Hand landmarks: 1,900/1,900 JSON outputs generated with MediaPipe Hand Landmarker in the remote `v2v_train` environment. The supplement was sampled at 12 FPS on CPU.
- Full landmark validation: 1,900/1,900 valid; every detected hand has 21 finite `[x,y,z]` points, timestamps are monotonic, and `hand_count` is consistent.

## Remote layout

```text
dataset_h3_capture_v1/
├── clips/                                      # 1,900 H.264 clips
├── manifests/capture_hand_detection_full.jsonl # 1,900 rows
├── manifests/source_inventory.jsonl             # 282 source rows
├── manifests/clips_full.sha256
├── reports/capture_full_validation.json
├── reports/source_coverage_full.json
├── reports/hand_keypoints_full_merge_validation.json
└── hand_keypoints/                              # canonical JSON plus traceable shards
```

Canonical keypoint files are `hand_keypoints/*_hands.json`. `flat/`,
`shard_00..08/`, and `supplement/` contain hard links or trace copies for
traceability; they must not be counted as additional samples.

## Interpretation boundary

The 1,882 base windows are motion-based high-recall candidates, and the 7
coverage-only windows are full-source fallbacks. They are not 1,900 confirmed
independent semantic actions. The 11 short-source clips are retained for
landmark coverage but are below the normal 4-second training-window rule.
The manifest remains `split=unassigned`; manual review must add object/action,
response/causality, occlusion, and train/validation/test split labels before
final H3 training.

H3 model training has not been started. This processing run did not occupy a
GPU; MediaPipe detection ran on remote CPU.
