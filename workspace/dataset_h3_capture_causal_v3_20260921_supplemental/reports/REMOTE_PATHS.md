# v3 远端路径

原片（只读）：

`/mnt/highspeed/users/nina/dataset_h3_capture_causal_full_20260920/sources`

第一批低重叠切片：

`/mnt/highspeed/users/nina/dataset_h3_capture_causal_v3_20260921/batches/primary_exportable/clips`

补充切片和语义更正：

`/nvmedata/workspace2/users/nina/dataset_h3_capture_causal_v3_20260921_supplemental/batches/seed_overlap/clips`

`/nvmedata/workspace2/users/nina/dataset_h3_capture_causal_v3_20260921_supplemental/batches/auto_overlap/clips`

`/nvmedata/workspace2/users/nina/dataset_h3_capture_causal_v3_20260921_supplemental/batches/semantic_corrections/clips`

每个批次的 `manifests/export_manifest.jsonl` 是视频与源时间的对应关系，`reports/validation.json` 是技术校验结果。所有视频均保持 `pending_user_review`；目前没有关键点 JSON，也没有启动训练。
