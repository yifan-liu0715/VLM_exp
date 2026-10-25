# Wan2.2-TI2V-5B Paired V2V LoRA — Formal Training Completion Report

## Outcome

- Status: completed normally (`max_epochs=5` reached)
- Training window: 2026-07-15 12:58:03–16:35:13 CST (about 3 h 37 min)
- Inference/validation generation: not run in this phase
- Physical training GPUs: 1, 2, 3; GPU 0 was excluded by `CUDA_VISIBLE_DEVICES=1,2,3`
- Optimizer steps: 3,265
- All 3,265 recorded step losses are finite

## Data

- Frozen manifest: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/metadata/metadata_full_paired_0715.csv`
- Total paired samples: 3,916
- Source splits included by the requested all-data policy: train 2,993; val 468; test 455
- Resolution buckets: 1280x512 = 1,032; 1280x704 = 1,225; 704x1280 = 1,659
- Every row has an existing `input_video`, `output_video`, and `reference_image`, plus a non-empty caption
- Unique sample IDs: 3,916; duplicates: 0
- Training mapping: `input_video + reference_image + caption -> output_video`
- Frames per sample: 33

Because all original splits were intentionally used for training, later unbiased evaluation must use a separate held-out dataset that is absent from this manifest.

## Model and training configuration

- Base model: Wan2.2-TI2V-5B
- Download source used for this formal run: Hugging Face directory `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/pretrain_weights/Wan2.2-TI2V-5B-HF`
- Converted DiT used by training: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/base_components/transformer`
- LoRA rank/alpha/dropout: 64 / 64 / 0.0
- LoRA targets: `to_q`, `to_k`, `to_v`, `to_out.0`
- Additional trainable source-video branch: `patch_embedding_extra`
- Trainable parameters: 94,964,736
- Batch size per GPU: 1
- Gradient accumulation: 2
- Effective batch size: 6 across three GPUs
- Epochs: 5
- Optimizer: AdamW; peak learning rate 1e-4; warmup 250; cosine decay
- Precision: BF16 mixed precision
- Gradient checkpointing: enabled
- Seed: 42

Authoritative configuration: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/configs/train_v2v_lora_0715.yaml`

## Training metrics

| Epoch | Mean training loss |
|---:|---:|
| 1 | 0.16220655 |
| 2 | 0.13850681 |
| 3 | 0.13770853 |
| 4 | 0.13479167 |
| 5 | 0.14697656 |

- Final recorded step loss: 0.11179177
- Last-20-step mean loss: 0.14000592
- Last-100-step mean loss: 0.15517113
- No Traceback, CUDA OOM, NCCL error, NaN/Inf loss, killed-process, or no-space error was found in the formal log.

Loss alone proves optimization ran successfully; it does not prove visual-quality improvement. Visual and semantic evaluation is deliberately deferred until a separate inference phase.

## Final weights

Preferred final adapter:

`/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/final_adapter/v2v_lora_and_source_patch.safetensors`

- File size: 189,991,384 bytes
- Tensor payload: 189,929,472 bytes
- Tensor count: 482 BF16 tensors
- Composition: 480 LoRA tensors + 2 `patch_embedding_extra` tensors
- All tensors finite and all 482 tensors contain nonzero values
- Saved global step: 3,265; saved epoch: 5
- SHA256: `8b4b2b5cdef9c4edce622519d5b4b6f4d894232776265e00d8931342ecd62db1`

Recovery checkpoint (step 3,000, not the preferred final result):

`/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/checkpoints/latest_v2v_lora_and_source_patch.safetensors`

- SHA256: `c84904ec618f0a5578cd37a1c33e1f4f3ae27a922caf4d12fffc09171a6f02de`

## Reproducibility hashes

- Training config: `48dfac49362f8c4da8dea121573a21ad36cee47a98d8d4b22050c23b14c93114`
- Frozen data manifest: `4fd0a1cd72b263c20bbc9b66870ba8883125f10ff5817ed94cca16fb9a02a3be`
- Formal training log: `4371052d3f29d7a5b72a01e00d5a62819b10cb42900c800a21500d3f1a449c7d`
- Lightning metrics CSV: `0cf4ec7520f33aa9f2db695060bbf991049f397654e970845317c5dc8d8240eb`

## Authoritative artifacts

- Formal log: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/logs/train_formal_gpu123.log`
- Step metrics: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/logs/lightning/version_0/metrics.csv`
- Run manifest: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/reports/run_manifest.json`
- Preflight report: `/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_formal_0715/reports/preflight_report.json`

