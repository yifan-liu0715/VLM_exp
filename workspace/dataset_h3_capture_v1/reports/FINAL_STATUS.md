# Capture hand-keypoint dataset status

Updated: 2026-09-18

## Current dataset

- Source directory: `G:\论文数据集\raw_data\DCIM\DJI_001`.
- Source inventory: 336 videos, all readable by `ffprobe`.
- Previous batch: 282 source videos and 1,900 exported clips.
- 2026-09-18 increment: 54 new source videos and 109 exported clips.
- Current remote total: 2,009 clips covering 336/336 source videos.
- Original source files were read only; none were moved, overwritten, or modified.

## Increment processing

- New-source audit: 54/54 readable, all 1920x1080 HEVC at 30000/1001 FPS.
- Candidate selection: 107 motion/grid candidates plus 2 full-source coverage fallbacks.
- Export: 109/109 valid H.264 MP4 clips, 24 FPS, no audio, 1280x720.
- Export duration range: 4.125 to 5.0 seconds.
- Remote upload SHA-256: 109/109 matched.
- MediaPipe detection: 109/109 completed at 24 FPS in the remote `v2v_train` environment with `CUDA_VISIBLE_DEVICES` empty; landmark inference used the TensorFlow Lite XNNPACK CPU delegate.
- Increment keypoint validation: 109/109 valid, with no missing clip, hash mismatch, malformed JSON, empty frame list, or landmark-structure error.
- Hand visibility review: 107/109 increment clips contain at least one detected hand. `capture_001930` and `capture_001939` are structurally valid but contain no detected hand in any sampled frame; keep them out of the training split unless manual review finds a labeling reason to retain them.

## Full validation

- Full manifest: 2,009 rows and 2,009 unique, continuous IDs (`capture_000001` through `capture_002009`).
- Canonical clips: 2,009/2,009 present, with matching SHA-256 values.
- Canonical keypoint JSON: 2,009/2,009 present and structurally valid.
- Source coverage: 336/336, with no uncovered or unexpected source.
- Coverage kinds: 1,882 `base_auto`, 107 `increment_auto`, 9 `coverage_only`, and 11 `short_source`.
- The 11 `short_source` clips came from the previous batch and remain marked below the normal 4-second training-window rule. The new increment contains no short-source clips.

## Remote layout

```text
/nvmedata/workspace2/users/nina/dataset_h3_capture_v1/
|-- clips/                                      # 2,009 H.264 clips
|-- manifests/capture_hand_detection_full.jsonl # 2,009 rows
|-- manifests/source_inventory.jsonl             # 336 sources
|-- manifests/clips_full.sha256                   # 2,009 hashes
|-- manifests/archive/20260918_increment_01/      # preserved 1,900-row baseline
|-- hand_keypoints/                               # 2,009 canonical JSON files
|-- hand_keypoints/shard_09/                      # 109 new JSON files
`-- reports/capture_increment_20260918_increment_01/
```

The `flat/` and `shard_*` paths contain hard links for traceability and must not be counted as additional samples.

## Training boundary

The 2,009 clips are high-recall action candidates, not 2,009 manually confirmed semantic actions. Object/action labels, response and causality labels, occlusion review, and train/validation/test splits still need manual review before model training.

H3 model training was not started. This increment did not reserve a GPU.
