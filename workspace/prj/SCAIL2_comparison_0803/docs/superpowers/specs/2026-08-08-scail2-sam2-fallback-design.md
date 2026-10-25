# SCAIL-2 Base/DPO Comparison with a SAM2 Fallback

Date: 2026-08-08

## Objective

Run a controlled SCAIL-2 Base-versus-DPO inference comparison on the same three test samples previously fixed for Scheme4, after Meta rejected access to the gated SAM3 checkpoint.

This experiment substitutes the official SAM3 mask preprocessor with the official open Meta SAM2 implementation. It is an ablation of the preprocessing path, not a claim of exact official SCAIL-2 reproduction.

## Scope

The first gate uses only `mymotion__humandance__batch1_mingxing_cropped__002`, because it has one prominent actor and makes mask identity switches easy to detect. If the mask and both generated videos pass the gate, the same frozen pipeline is applied to:

- `mymotion__051`
- `mymotion__single_talking_face__datatrain__002`
- `mymotion__humandance__batch1_mingxing_cropped__002`

No SCAIL-2 training or fine-tuning is performed. The existing converted Base and DPO weights are inference-only inputs.

## Non-goals

- Do not claim an official SAM3-based SCAIL-2 reproduction.
- Do not use target-video pixels, target masks, or target-derived boxes as generation conditions.
- Do not use all-white, empty, or fabricated masks merely to bypass input validation.
- Do not compare against Scheme4 in this phase.
- Do not use physical GPU0.

## Isolation and Paths

Experiment root:

`/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803`

New fallback assets are isolated under:

- Code and manifests: `sam2_fallback_0808/`
- Official SAM2 weights: `weights/sam2_fallback/`
- Preprocessed sample inputs: `preprocessed_sam2/<sample_id>/`
- Base/DPO outputs: `outputs/sam2_fallback_base_vs_dpo/<sample_id>/`
- Mask environment: `/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808`

The existing SCAIL-2 environment and converted weights remain unchanged.

## Inputs and Leakage Boundary

For each sample:

- Driving/source video: `output_video` from `manifests/input_manifest.json`
- Reference image: `reference` from `manifests/input_manifest.json`
- Prompt: the frozen prompt in `manifests/input_manifest.json`
- Target video: `input_video`, used only after generation for visual evaluation

Preprocessing scripts must reject target-video paths in their command-line arguments and manifests. Tests must verify that no target path appears in the preprocessing command or output provenance.

## Mask Preprocessing Architecture

1. Pin an exact official `facebookresearch/sam2` source commit and record its archive/repository SHA256.
2. Download an official SAM2.1 checkpoint from the Meta repository and record its SHA256.
3. Use the person detector distributed through the official DWPose/SCAIL-Pose instructions to obtain candidate person boxes on the reference image and first source frame.
4. Segment the reference actor with the SAM2 image predictor.
5. Initialize the SAM2 video predictor with the selected source-frame box or mask and propagate the actor mask through all 97 source frames.
6. For multi-person inputs, select the source track using the same spatial principle described by SCAIL-Pose `matchnearest`: resize the reference mask to the source canvas and choose the first-frame track with the greatest mask IoU. Record the selected track and all candidate scores.
7. Encode reference and video masks with the exact palette and value convention expected by the pinned SCAIL-2 WAN code.
8. In end-to-end driving mode, copy the source video to `rendered_v2.mp4`; write the propagated colored mask video to `rendered_mask_v2.mp4`; write the reference mask to `ref_mask.jpg`.

No target frame is used in detection, track selection, segmentation, or mask correction.

## Mask Quality Gate

Before any SCAIL-2 inference, the single dance sample must satisfy all checks:

- 97 mask frames, 24 fps, and the same width and height as the source video.
- A non-empty selected actor mask on every frame.
- No actor-ID switch on manually inspected frames 0, 48, and 96.
- No unexplained single-frame mask-area jump greater than 50% relative to the previous frame.
- Reference mask contains one valid foreground component after removing tiny disconnected components.
- Side-by-side overlays for frames 0, 48, and 96 show that hair, body, hands, and clothing are assigned to the intended actor without absorbing large background regions.

If automatic detection selects the wrong actor, one manually recorded initial bounding box is allowed. It must be stored in the run manifest. Frame-by-frame manual masks are outside this first phase.

## Base/DPO Inference Contract

Base and DPO use identical:

- source/driving video
- reference image and mask
- source mask video
- prompt
- native source width and height
- `replace_flag`
- solver `unipc`
- 40 sampling steps
- shift 3.0
- CFG 5.0
- seed 42
- segment length 81
- segment overlap 5
- model offload setting

DPO differs only by:

- `--lora_path weights/converted/bias-aware-dpo-lora.safetensors`
- `--lora_alpha 1.0`

The existing static inference-plan test remains the authority for this equality constraint.

## Execution Gates

1. CPU-only source, checkpoint, configuration, and input-manifest audits.
2. Mask preprocessing smoke test on an idle physical GPU other than GPU0.
3. Mask quality gate on the dance sample.
4. Base inference on the dance sample.
5. DPO inference using the exact same inputs.
6. Video validation and multi-frame inspection.
7. Expand to the other two samples only if the dance sample is valid.

Each long task runs in an isolated background session with low-frequency monitoring. A new GPU availability check is required before every GPU stage.

## Validation and Evidence

Each run records:

- pinned source commits
- checkpoint paths, sizes, and SHA256 values
- source/reference/prompt provenance
- all detector candidates and selected track
- mask statistics and frame-level warnings
- complete inference arguments
- physical GPU assignment
- log path, exit code, and elapsed time
- output video `ffprobe` report

The final visual deliverable for each successful sample contains Source, Reference, Target (evaluation only), Base, and DPO views with labels that explicitly state `SAM2 fallback masks`.

The first conclusion is limited to whether DPO improves identity, texture, motion consistency, and artifacts relative to Base under the same SAM2-derived conditions. It does not establish equivalence to official SAM3 preprocessing.

## Failure Handling

- If official SAM2 assets cannot be downloaded or verified, stop before installation.
- If SAM2 propagation loses the actor, preserve logs and overlays; try one recorded manual initial box.
- If the manual-box retry still fails, stop the sample and propose YOLO segmentation plus tracking as a separate experiment rather than silently changing the method.
- If inference fails with OOM, preserve logs and report evidence. Do not use GPU0 or silently change resolution, frame count, or sampling parameters.
- Do not delete shared files or overwrite previous SCAIL-2/Scheme4 results.

## Completion Criteria

The phase is complete when the dance sample has valid SAM2-derived conditioning, valid Base and DPO videos, reproducible manifests, and a defensible Base-versus-DPO conclusion. Expansion to all three samples is completed only after the single-sample gate passes.
