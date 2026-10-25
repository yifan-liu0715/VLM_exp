# Wan V2V Reference Geometry Scheme 1 Design

## Goal

Run an isolated 2,000-step diagnostic training in which Source and Target
preprocessing remain byte-for-byte equivalent to the completed teacher-aligned
experiment, while the external Reference image is converted to the same final
frame size without non-uniform stretching.

## Scope

- Use the existing corrected manifests and physical media under
  `/nvmedata/workspace2/users/nina/dataset_video_train`.
- Keep Source/Target frame sampling, resize, VAE encoding, Flow Matching loss,
  LoRA rank/alpha, optimizer, dropout, seed, and scheduler unchanged.
- Replace only the Reference preprocessing.
- Use the same Reference preprocessing implementation in training and inference.
- Train for exactly 2,000 global steps.
- Never use physical GPU 0.
- Do not create or run Scheme 2.

## Reference transform

The output tensor must have the same `height,width` as the processed
Source/Target frame. The transform must never resize width and height with
different scale factors.

1. Read the original `1024x1024` RGB Reference.
2. Estimate whether it has a predominantly white border.
3. For a white-background portrait, find the non-white foreground bounds and
   add a 12% margin.
4. If a crop with the requested output aspect ratio can retain that foreground,
   crop around the foreground center and resize uniformly.
5. If such a crop would remove foreground, uniformly fit the foreground region
   inside the requested canvas and fill the remaining area with white.
6. For a non-white-background image, use an aspect-preserving centered crop.
7. Normalize to `[-1,1]` and return `C,H,W`.

This preserves the model's existing equal-shape contract:

```text
Source HxW = Target HxW = Reference HxW
```

while removing Reference-only geometric stretching.

## Experiment isolation

Remote root:

```text
/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/v2v_reference_aspect_scheme1_0727
```

Local result root:

```text
D:\Code\Wan2.2_v2v_reference_aspect_scheme1_<timestamp>
```

The remote root contains copied scripts, frozen manifests, config, tests, logs,
checkpoints, final adapter, validation videos, and an experiment manifest.

## Validation

Before training:

- Unit tests prove output shape equality and aspect preservation on synthetic
  references.
- Contract tests prove Source/Target preprocessing source is unchanged.
- Training and inference import the same Reference transform.
- A CPU dataset check loads one sample from each native resolution.
- A one-step non-GPU0 model preflight must produce a finite loss.

After training:

- Infer one held-out sample from each native resolution.
- Use identical prompt, Source, Reference, seed 42, 50 steps, CFG 5, and Flow
  shift 5 for Base and Scheme-1 LoRA.
- Save Source, Target, Base, Scheme-1 LoRA, and a four-panel comparison.
- Verify every MP4 with `ffprobe` and inspect multiple motion frames.

## Interpretation limit

This 2,000-step run is a pipeline diagnostic, not a quality-equivalent
comparison with the existing 20,000-step adapter. It can reveal whether
Reference geometry is healthier, but it cannot establish the final achievable
quality without a matched longer run.
