# SCAIL-2 SAM2 Fallback Base/DPO Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, target-leakage-free SAM2 mask path and use it to compare the converted SCAIL-2 Base and DPO LoRA under identical inference conditions.

**Architecture:** A new isolated `sam2_fallback_0808` package owns asset provenance, manifest loading, YOLOX person detection, SAM2 segmentation/tracking, SCAIL-2 palette conversion, and mask QC. The existing pinned WAN generator stays unchanged; its static plan is revised to accept only approved SAM2 bundles, with the dance sample acting as a gate before three-sample expansion.

**Tech Stack:** Python 3.11, PyTorch 2.9.1+cu126, official Meta SAM2.1 Hiera Large, official DWPose YOLOX-L ONNX, ONNX Runtime, OpenCV, Pillow, NumPy 1.26.4, FFmpeg/ffprobe, SCAIL-2 WAN commit `5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5`, unittest, Git, SHA256.

## Global Constraints

- Remote host: `student@115.190.111.129`.
- Experiment root: `/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803`.
- SCAIL environment: `/nvmedata/workspace2/users/nina/conda_envs/scail2_0803`.
- Mask environment: `/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808`.
- Never use physical GPU0. Record and verify the selected physical GPU before every GPU launch.
- Stop new writes below 30 GB free on `/nvmedata`; preserve logs and never delete shared files.
- Do not modify converted Base/DPO weights, Scheme4 assets, or pinned official source snapshots.
- Dataset `input_video` is evaluation-only and must never enter detection, segmentation, tracking, masks, or generation conditioning.
- Gate first on `mymotion__humandance__batch1_mingxing_cropped__002`; expand only after mask and Base/DPO output QC pass.
- Base/DPO share source, reference, masks, prompt, native size, `unipc`, 40 steps, shift 3.0, CFG 5.0, seed 42, segment length 81, overlap 5, replacement mode, and offload.
- DPO differs only by `--lora_path weights/converted/bias-aware-dpo-lora.safetensors` and `--lora_alpha 1.0`.
- Every artifact says `SAM2 fallback masks`; this is not an official SAM3 SCAIL-2 reproduction.
- No SCAIL-2 training or fine-tuning occurs.

## File Structure

- `sam2_fallback_0808/assets.py`: pinned assets and hash validation.
- `sam2_fallback_0808/inputs.py`: manifest mapping and leakage guard.
- `sam2_fallback_0808/detector.py`: YOLOX person boxes and actor selection.
- `sam2_fallback_0808/segment.py`: SAM2 image/video predictor adapters.
- `sam2_fallback_0808/palette.py`: reference/video mask color encoding.
- `sam2_fallback_0808/quality.py`: mask statistics and gate decision.
- `scripts/fetch_sam2_fallback_assets.py`: official source/checkpoint acquisition.
- `scripts/prepare_sam2_masks.py`: one-sample preprocessing.
- `scripts/validate_sam2_masks.py`: ffprobe, overlays, and QC manifest.
- `scripts/select_idle_nonzero_gpu.py`: non-GPU0 selection.
- `scripts/build_scail2_inference_plan.py`: existing Base/DPO plan revised for SAM2.
- `scripts/run_scail2_sam2_gate.py`: fail-closed Base/DPO controller.
- `scripts/build_scail2_sam2_report.py`: grids, hashes, and final report.

---

### Task 1: Pin and Verify Official Assets

**Files:**
- Create: `sam2_fallback_0808/__init__.py`
- Create: `sam2_fallback_0808/assets.py`
- Create: `scripts/fetch_sam2_fallback_assets.py`
- Test: `tests/test_sam2_assets.py`
- Create after execution: `manifests/sam2_fallback_assets.json`

**Interfaces:**
- Produces: `AssetSpec(name, url, path, expected_sha256)`.
- Produces: `sha256_file(Path) -> str`, `verify_asset(Path, AssetSpec) -> dict`.
- Produces source snapshots `code/sam2-2b90b9f5ceec907a1c18123530e92e794ad901a4` and `code/DWPose-3dca5db79d9f9ffdd378753ddf6ec66535aace88`.
- Produces checkpoints `weights/sam2_fallback/sam2.1_hiera_large.pt` and `weights/sam2_fallback/yolox_l.onnx`.

- [ ] **Step 1: Write the failing test**

```python
import hashlib, tempfile, unittest
from pathlib import Path
from sam2_fallback_0808.assets import AssetSpec, sha256_file, verify_asset

class AssetTest(unittest.TestCase):
    def test_expected_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset.bin"
            path.write_bytes(b"official")
            digest = hashlib.sha256(b"official").hexdigest()
            spec = AssetSpec("fixture", "https://example.invalid/a", str(path), digest)
            self.assertEqual(sha256_file(path), digest)
            self.assertTrue(verify_asset(path, spec)["verified"])
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_asset(path, AssetSpec("fixture", spec.url, str(path), "0" * 64))
```

- [ ] **Step 2: Run the test to verify failure**

Run: `/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_sam2_assets -v`

Expected: import failure for `sam2_fallback_0808.assets`.

- [ ] **Step 3: Implement asset validation and the fetcher**

```python
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
DWPOSE_COMMIT = "3dca5db79d9f9ffdd378753ddf6ec66535aace88"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
YOLOX_URL = "https://huggingface.co/yzd-v/DWPose/resolve/f7c16a3d45ad3783db41471848c80fbc281cabac/yolox_l.onnx?download=true"
YOLOX_SHA256 = "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"

@dataclass(frozen=True)
class AssetSpec:
    name: str
    url: str
    path: str
    expected_sha256: str | None

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_asset(path: Path, spec: AssetSpec) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if spec.expected_sha256 and actual != spec.expected_sha256:
        raise ValueError(f"SHA256 mismatch for {spec.name}: {actual}")
    return {**asdict(spec), "size": path.stat().st_size, "sha256": actual, "verified": True}
```

The fetcher clones the two official repositories at the exact commits into the paths above, downloads the two checkpoint paths above, verifies YOLOX against the fixed SHA, records the computed SAM2 checkpoint SHA on first successful download, and atomically writes the asset manifest. Every later run loads that recorded digest as the SAM2 expected hash. `--verify-only` must never access the network.

- [ ] **Step 4: Run tests and static checks**

```bash
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_sam2_assets -v
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m py_compile sam2_fallback_0808/assets.py scripts/fetch_sam2_fallback_assets.py
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add sam2_fallback_0808/__init__.py sam2_fallback_0808/assets.py scripts/fetch_sam2_fallback_assets.py tests/test_sam2_assets.py
git commit -m "Add verified SAM2 fallback assets"
```

---

### Task 2: Create the Isolated Mask Environment

**Files:**
- Create: `scripts/create_sam2_mask_env.sh`
- Create after execution: `manifests/sam2_mask_conda_explicit.txt`
- Create after execution: `manifests/sam2_mask_pip_freeze.txt`

**Interfaces:**
- Consumes: Task 1 fetcher.
- Produces: `/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python` and verified assets.

- [ ] **Step 1: Write the environment script**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808
test "$(df --output=avail /nvmedata | tail -n 1)" -ge $((30 * 1024 * 1024))
test -x "$ENV/bin/python" || conda create -y -p "$ENV" --clone /nvmedata/workspace2/users/nina/conda_envs/scail2_0803
"$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV/bin/python" -m pip install onnxruntime-gpu hydra-core iopath tqdm
"$ENV/bin/python" -m pip install -e "$ROOT/code/sam2-2b90b9f5ceec907a1c18123530e92e794ad901a4"
"$ENV/bin/python" -m pip check
conda list -p "$ENV" --explicit > "$ROOT/manifests/sam2_mask_conda_explicit.txt"
"$ENV/bin/python" -m pip freeze > "$ROOT/manifests/sam2_mask_pip_freeze.txt"
```

- [ ] **Step 2: Fetch assets CPU-only**

```bash
cd /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
CUDA_VISIBLE_DEVICES='' /nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python \
  scripts/fetch_sam2_fallback_assets.py --root "$PWD" > logs/sam2_asset_fetch.log 2>&1
```

Expected: exact source commits, fixed YOLOX SHA, and non-empty computed SAM2 SHA appear in the manifest.

- [ ] **Step 3: Create and validate the environment**

```bash
bash scripts/create_sam2_mask_env.sh > logs/sam2_env_install.log 2>&1
/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python - <<'PY'
import cv2, numpy, onnxruntime, torch
from sam2.build_sam import build_sam2_video_predictor
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(cv2.__version__, numpy.__version__, onnxruntime.__version__)
print(build_sam2_video_predictor.__name__)
PY
```

Expected: imports succeed; this step does not allocate a model.

- [ ] **Step 4: Verify assets without downloading**

Run: `CUDA_VISIBLE_DEVICES='' /nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python scripts/fetch_sam2_fallback_assets.py --root "$PWD" --verify-only`

Expected: exit 0 and all assets verified.

- [ ] **Step 5: Commit small files only**

```bash
git add scripts/create_sam2_mask_env.sh manifests/sam2_fallback_assets.json manifests/sam2_mask_conda_explicit.txt manifests/sam2_mask_pip_freeze.txt
git commit -m "Record isolated SAM2 mask environment"
```

Do not add checkpoints, source clones, environments, or logs.

---

### Task 3: Enforce the Input Mapping and Leakage Boundary

**Files:**
- Create: `sam2_fallback_0808/inputs.py`
- Test: `tests/test_sam2_inputs.py`

**Interfaces:**
- Produces: immutable `SampleInput` and `load_sample(root, sample_id)`.
- Produces: `assert_conditioning_paths(sample, paths)`.

- [ ] **Step 1: Write the failing test**

```python
import json, tempfile, unittest
from pathlib import Path
from sam2_fallback_0808.inputs import assert_conditioning_paths, load_sample

class BoundaryTest(unittest.TestCase):
    def test_mapping_and_target_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "manifests").mkdir()
            row = {"sample_id":"dance", "source_video":"/d/output_video/dance.mp4",
                   "target_video":"/d/input_video/dance.mp4", "reference_image":"/d/reference/dance.jpg",
                   "prompt":"dance", "width":704, "height":1280, "num_frames":97, "fps":24.0}
            (root / "manifests/input_manifest.json").write_text(json.dumps({"samples":[row]}))
            sample = load_sample(root, "dance")
            assert_conditioning_paths(sample, [sample.source_video, sample.reference_image])
            with self.assertRaisesRegex(ValueError, "target leakage"):
                assert_conditioning_paths(sample, [sample.target_video])
```

- [ ] **Step 2: Verify failure**

Run: `/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_sam2_inputs -v`

Expected: import failure.

- [ ] **Step 3: Implement the boundary**

```python
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SampleInput:
    sample_id: str; source_video: Path; reference_image: Path; prompt: str
    width: int; height: int; num_frames: int; fps: float; target_video: Path

def load_sample(root: Path, sample_id: str) -> SampleInput:
    rows = json.loads((root / "manifests/input_manifest.json").read_text())["samples"]
    row = next(item for item in rows if item["sample_id"] == sample_id)
    sample = SampleInput(row["sample_id"], Path(row["source_video"]), Path(row["reference_image"]),
                         row["prompt"], int(row["width"]), int(row["height"]),
                         int(row["num_frames"]), float(row["fps"]), Path(row["target_video"]))
    if "output_video" not in str(sample.source_video) or "input_video" not in str(sample.target_video):
        raise ValueError("approved dataset direction violated")
    return sample

def assert_conditioning_paths(sample: SampleInput, paths) -> None:
    target = sample.target_video.resolve(strict=False)
    for path in paths:
        if Path(path).resolve(strict=False) == target:
            raise ValueError(f"target leakage: {path}")
```

- [ ] **Step 4: Test real manifests**

Run the unit test, then load all three fixed sample IDs and call `assert_conditioning_paths` with only source/reference. Expected: 1280x512, 1280x704, and 704x1280; each has 97 frames at 24 fps.

- [ ] **Step 5: Commit**

```bash
git add sam2_fallback_0808/inputs.py tests/test_sam2_inputs.py
git commit -m "Enforce SCAIL2 conditioning boundary"
```

---

### Task 4: Add YOLOX Person Detection and Actor Selection

**Files:**
- Create: `sam2_fallback_0808/detector.py`
- Test: `tests/test_sam2_detector.py`

**Interfaces:**
- Produces: `Detection(box_xyxy, score, class_id)`.
- Produces: `detect_people(image_rgb, model_path, score_threshold=0.30)`.
- Produces: `select_single_or_manual(detections, canvas, manual_box=None)`; multi-person matching is deferred to Task 5 mask IoU.

- [ ] **Step 1: Write failing deterministic-selection tests**

```python
import unittest
from sam2_fallback_0808.detector import Detection, select_single_or_manual

class DetectorTest(unittest.TestCase):
    def test_single_person_is_selected(self):
        actor = Detection((200, 100, 600, 1100), 0.88, 0)
        self.assertEqual(select_single_or_manual([actor], (704, 1280)), actor)
    def test_multiple_people_require_mask_iou_matching(self):
        people = [Detection((0,0,100,100),0.95,0), Detection((200,100,600,1100),0.88,0)]
        with self.assertRaisesRegex(ValueError, "mask IoU"):
            select_single_or_manual(people, (704, 1280))
    def test_manual_box_is_bounded_and_recorded(self):
        chosen = select_single_or_manual([], (704, 1280), (100, 50, 600, 1200))
        self.assertEqual(chosen.class_id, -1)
    def test_no_person_fails(self):
        with self.assertRaisesRegex(ValueError, "no person"):
            select_single_or_manual([], (704, 1280))
```

- [ ] **Step 2: Verify failure**

Run: `/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python -m unittest tests.test_sam2_detector -v`

- [ ] **Step 3: Implement selection and the official YOLOX wrapper**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int

def select_single_or_manual(detections, canvas, manual_box=None):
    width, height = canvas
    if manual_box is not None:
        x1, y1, x2, y2 = map(float, manual_box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("manual box outside source canvas")
        return Detection((x1, y1, x2, y2), 1.0, -1)
    people = [d for d in detections if d.class_id == 0]
    if not people:
        raise ValueError("no person detection")
    if len(people) > 1:
        raise ValueError("multiple people require first-frame mask IoU matching")
    return people[0]
```

`detect_people` uses YOLOX preprocessing/postprocessing from the pinned official DWPose commit with attribution, returns finite in-canvas XYXY boxes, NMS IoU 0.45, class 0, and score at least 0.30. It receives only reference/source pixels.

- [ ] **Step 4: Run tests and a direct CPU detector smoke**

Run tests, then invoke `detect_people` directly on the dance reference and source frame 0 from a short Python command that obtains both paths through `load_sample`. Expected: at least one finite class-0 box for each image. Save the two returned lists to `preprocessed_sam2/mymotion__humandance__batch1_mingxing_cropped__002/detector_candidates.json`; assert that serialized text does not contain the target path. Task 5 later integrates the same function into the full CLI.

- [ ] **Step 5: Commit**

```bash
git add sam2_fallback_0808/detector.py tests/test_sam2_detector.py
git commit -m "Add deterministic fallback actor detection"
```

---

### Task 5: Implement SAM2 Reference Segmentation and Video Propagation

**Files:**
- Create: `sam2_fallback_0808/segment.py`
- Create: `scripts/prepare_sam2_masks.py`
- Test: `tests/test_sam2_segment.py`

**Interfaces:**
- Consumes: `SampleInput`, `Detection`, verified SAM2 checkpoint, config `configs/sam2.1/sam2.1_hiera_l.yaml`.
- Produces: `segment_reference(predictor, image_rgb, box_xyxy) -> bool[H,W]`.
- Produces: `validate_propagated_masks(masks, expected_frames, expected_shape)`.
- Produces first: `preprocessed_sam2/mymotion__humandance__batch1_mingxing_cropped__002/raw_masks/reference.png` and 97 video PNGs.
- Produces: `select_source_track(reference_mask, candidate_masks) -> tuple[int, list[float]]` for later multi-person samples.

- [ ] **Step 1: Write failing predictor-adapter tests**

```python
import unittest
import numpy as np
from sam2_fallback_0808.segment import segment_reference, select_source_track, validate_propagated_masks

class FakePredictor:
    def set_image(self, image): self.shape = image.shape[:2]
    def predict(self, box, multimask_output):
        masks = np.zeros((1, *self.shape), dtype=bool)
        x1, y1, x2, y2 = box[0].astype(int)
        masks[0, y1:y2, x1:x2] = True
        return masks, np.array([0.9]), np.array([0.0])

class SegmentTest(unittest.TestCase):
    def test_reference_shape_and_dtype(self):
        mask = segment_reference(FakePredictor(), np.zeros((128,64,3), np.uint8), (5,10,50,100))
        self.assertEqual(mask.shape, (128,64)); self.assertEqual(mask.dtype, np.bool_)
    def test_all_97_frames_are_required(self):
        masks = [np.ones((8,8), bool) for _ in range(97)]
        validate_propagated_masks(masks, 97, (8,8))
        masks[48][:] = False
        with self.assertRaisesRegex(ValueError, "empty mask at frame 48"):
            validate_propagated_masks(masks, 97, (8,8))
    def test_multi_person_track_uses_highest_reference_iou(self):
        ref = np.zeros((8,8), bool); ref[2:6,2:6] = True
        wrong = np.zeros((8,8), bool); wrong[:2,:2] = True
        right = ref.copy()
        index, scores = select_source_track(ref, [wrong, right])
        self.assertEqual(index, 1); self.assertGreater(scores[1], scores[0])
```

- [ ] **Step 2: Verify failure**

Run: `/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python -m unittest tests.test_sam2_segment -v`

Expected: import failure.

- [ ] **Step 3: Implement adapters and CLI**

```python
import cv2
import numpy as np

def segment_reference(predictor, image_rgb, box_xyxy):
    predictor.set_image(image_rgb)
    box = np.asarray(box_xyxy, dtype=np.float32)[None, :]
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    mask = np.asarray(masks[int(np.argmax(scores))], dtype=bool)
    if mask.shape != image_rgb.shape[:2] or not mask.any():
        raise ValueError("invalid reference mask")
    return mask

def validate_propagated_masks(masks, expected_frames, expected_shape):
    if len(masks) != expected_frames:
        raise ValueError(f"expected {expected_frames} masks, got {len(masks)}")
    for index, mask in enumerate(masks):
        if mask.shape != expected_shape:
            raise ValueError(f"shape mismatch at frame {index}: {mask.shape}")
        if not np.asarray(mask, dtype=bool).any():
            raise ValueError(f"empty mask at frame {index}")

def select_source_track(reference_mask, candidate_masks):
    ref = np.asarray(reference_mask, dtype=bool)
    scores = []
    for candidate in candidate_masks:
        mask = np.asarray(candidate, dtype=bool)
        resized = cv2.resize(ref.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        union = np.logical_or(resized, mask).sum()
        scores.append(float(np.logical_and(resized, mask).sum() / max(union, 1)))
    if not scores:
        raise ValueError("no source candidate masks")
    return int(np.argmax(scores)), scores
```

The CLI first asserts the decoded source frames and reference image both exactly match the manifest canvas; it does not stretch either input. It extracts exactly 97 source frames and builds the official image and video predictors with the same Hiera Large checkpoint. With multiple source people it segments every first-frame candidate, calls `select_source_track`, and records every IoU plus the selected index; this is the approved `matchnearest` spatial principle. It then adds the chosen frame-0 box with `add_new_points_or_box`, collects `propagate_in_video` logits by frame index, thresholds at 0.0, validates, and writes binary PNGs. Its provenance contains source/reference only; `--help` must have no target option.

- [ ] **Step 4: Run tests and argument audit**

```bash
/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python -m unittest tests.test_sam2_segment -v
CUDA_VISIBLE_DEVICES='' /nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python scripts/prepare_sam2_masks.py --help
```

Expected: tests pass; help includes `--root`, `--sample-id`, `--manual-box-file`, `--detect-only`, `--device`, and no target input.

- [ ] **Step 5: Commit**

```bash
git add sam2_fallback_0808/segment.py scripts/prepare_sam2_masks.py tests/test_sam2_segment.py
git commit -m "Add SAM2 reference and video propagation"
```

---

### Task 6: Encode SCAIL-2 Masks and Implement QC

**Files:**
- Create: `sam2_fallback_0808/palette.py`
- Create: `sam2_fallback_0808/quality.py`
- Create: `scripts/validate_sam2_masks.py`
- Test: `tests/test_sam2_palette_quality.py`

**Interfaces:**
- Produces: `encode_reference(mask)` with white background and blue actor.
- Produces: `encode_video_frame(mask)` with black background and blue actor.
- Produces: `compute_mask_stats(masks)`, `validate_bundle(sample, output_dir)`.

- [ ] **Step 1: Write failing palette and jump tests**

```python
import unittest, numpy as np
from sam2_fallback_0808.palette import encode_reference, encode_video_frame
from sam2_fallback_0808.quality import compute_mask_stats

class PaletteQualityTest(unittest.TestCase):
    def test_exact_colors(self):
        mask = np.array([[False, True]], bool)
        np.testing.assert_array_equal(encode_reference(mask)[0,0], [255,255,255])
        np.testing.assert_array_equal(encode_video_frame(mask)[0,0], [0,0,0])
        np.testing.assert_array_equal(encode_reference(mask)[0,1], [0,0,255])
        np.testing.assert_array_equal(encode_video_frame(mask)[0,1], [0,0,255])
    def test_area_jump_fails_automatic_gate(self):
        masks = [np.zeros((10,10), bool) for _ in range(3)]
        masks[0][:2,:2] = True; masks[1][:8,:8] = True; masks[2][:8,:8] = True
        report = compute_mask_stats(masks)
        self.assertEqual(report["jump_frames"], [1]); self.assertFalse(report["automatic_pass"])
```

- [ ] **Step 2: Verify failure**

Run: `/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python -m unittest tests.test_sam2_palette_quality -v`

- [ ] **Step 3: Implement palette and QC**

```python
import numpy as np

BLUE = np.array([0,0,255], np.uint8)
WHITE = np.array([255,255,255], np.uint8)
BLACK = np.array([0,0,0], np.uint8)

def encode_reference(mask):
    out = np.broadcast_to(WHITE, (*mask.shape,3)).copy(); out[np.asarray(mask,bool)] = BLUE
    return out

def encode_video_frame(mask):
    out = np.broadcast_to(BLACK, (*mask.shape,3)).copy(); out[np.asarray(mask,bool)] = BLUE
    return out
```

`compute_mask_stats` records pixel count/ratio for every frame and flags `i` when `abs(area[i]-area[i-1])/max(area[i-1],1) > 0.50`. `validate_bundle` requires 97 nonempty frames, 24 fps, source dimensions, removes reference components smaller than 0.1% of the canvas, requires exactly one remaining reference component, exports overlays 0/48/96, and requires `qc_review.json` exactly containing `status=passed`, frames `[0,48,96]`, `id_switch=false`, and `large_background_absorption=false`. Encode `rendered_mask_v2.mp4` with `libx264 -crf 0 -pix_fmt yuv444p`; copy source bytes to `rendered_v2.mp4`.

- [ ] **Step 4: Test codec round-trip**

```bash
/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python -m unittest tests.test_sam2_palette_quality -v
/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python scripts/validate_sam2_masks.py --self-test-codec
```

Expected: decoded foreground remains blue-dominant (`B>200,R<64,G<64`) and video background black.

- [ ] **Step 5: Commit**

```bash
git add sam2_fallback_0808/palette.py sam2_fallback_0808/quality.py scripts/validate_sam2_masks.py tests/test_sam2_palette_quality.py
git commit -m "Add SCAIL2 mask palette and QC gate"
```

---

### Task 7: Run the Dance Mask Gate on a Nonzero GPU

**Files:**
- Create: `scripts/select_idle_nonzero_gpu.py`
- Test: `tests/test_select_idle_nonzero_gpu.py`
- Create after execution: `preprocessed_sam2/mymotion__humandance__batch1_mingxing_cropped__002/{ref_mask.jpg,rendered_v2.mp4,rendered_mask_v2.mp4,mask_manifest.json,qc_review.json}`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: the first approved mask bundle.

- [ ] **Step 1: Test GPU0 exclusion**

Write `select_gpu(rows, minimum_free_mib) -> int`; exclude index 0, require utilization at most 10%, then maximize free memory. Test data must include idle GPU0 and assert it is never returned.

Run: `/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_select_idle_nonzero_gpu -v`

- [ ] **Step 2: Select and record a GPU**

```bash
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
GPU=$(/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python "$ROOT/scripts/select_idle_nonzero_gpu.py" --minimum-free-mib 16000)
test "$GPU" != 0
printf '%s\n' "$GPU" > "$ROOT/logs/sam2_mask_dance.physical_gpu"
```

- [ ] **Step 3: Launch isolated preprocessing**

```bash
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
GPU=$(cat "$ROOT/logs/sam2_mask_dance.physical_gpu")
tmux new-session -d -s scail2_sam2_mask_dance_0808 \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES='$GPU' /nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python scripts/prepare_sam2_masks.py --root '$ROOT' --sample-id mymotion__humandance__batch1_mingxing_cropped__002 --device cuda:0 > logs/sam2_mask_dance.log 2>&1; printf '%s' \$? > logs/sam2_mask_dance.exit"
```

Monitor at low frequency and verify the process is bound only to the recorded physical GPU.

- [ ] **Step 4: Validate and review frames 0/48/96**

Run `validate_sam2_masks.py` for the dance sample. If automatic detection selects the wrong actor, archive the failed manifest; use the validator's rectangle UI on `source_frame_00000.jpg` to write `manual_box.json`, then permit one retry via `--manual-box-file manual_box.json`. Do not create per-frame manual masks. Write `qc_review.json` only after checking identity continuity and hair/body/hands/clothing/background coverage. If this retry still fails, stop and propose YOLO segmentation plus tracking as a separately labeled experiment.

- [ ] **Step 5: Revalidate and commit compact evidence**

```bash
/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808/bin/python scripts/validate_sam2_masks.py --root /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803 --sample-id mymotion__humandance__batch1_mingxing_cropped__002 --require-human-review
git add scripts/select_idle_nonzero_gpu.py tests/test_select_idle_nonzero_gpu.py preprocessed_sam2/mymotion__humandance__batch1_mingxing_cropped__002/mask_manifest.json preprocessed_sam2/mymotion__humandance__batch1_mingxing_cropped__002/qc_review.json
git commit -m "Approve dance SAM2 fallback mask gate"
```

Expected: status `passed`. Do not commit videos, images, raw masks, checkpoints, or logs.

---

### Task 8: Revise the Static Base/DPO Plan for SAM2

**Files:**
- Modify: `scripts/build_scail2_inference_plan.py`
- Modify: `tests/test_build_scail2_inference_plan.py`
- Create after execution: `manifests/inference_plan_sam2.json`

**Interfaces:**
- Consumes: passed `mask_manifest.json` and three condition files.
- Produces: six runs with readiness per sample; preserves Base/DPO equality.

- [ ] **Step 1: Add failing SAM2 readiness tests**

Create a passed mask fixture and assert:

```python
self.assertEqual(plan["mask_preprocessor"], "official SAM2 fallback; not official SAM3")
self.assertTrue(base["ready"])
self.assertIn("preprocessed_sam2", option_map(base["argv"])["--mask_image"])
self.assertEqual(option_map(base["argv"])["--mask_video"], option_map(dpo["argv"])["--mask_video"])
```

Create the same files with manifest status `failed` and assert both variants remain unready.

- [ ] **Step 2: Verify failure**

Run: `/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_build_scail2_inference_plan -v`

Expected: old `preprocessed/` path and SAM3 status cause failure.

- [ ] **Step 3: Implement readiness**

```python
def approved_mask_bundle(root: Path, sample_id: str) -> bool:
    processed = root / "preprocessed_sam2" / sample_id
    manifest_path = processed / "mask_manifest.json"
    if not manifest_path.is_file(): return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = ("ref_mask.jpg", "rendered_v2.mp4", "rendered_mask_v2.mp4")
    return manifest.get("status") == "passed" and all((processed / n).is_file() for n in names)
```

Point common arguments to `preprocessed_sam2`, outputs to `outputs/sam2_fallback_base_vs_dpo`, and add the preprocessor label. Preserve all sampling values and the existing DPO-only-difference test.

- [ ] **Step 4: Test and build real plan**

```bash
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_build_scail2_inference_plan -v
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python scripts/build_scail2_inference_plan.py --root /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803 --output /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/inference_plan_sam2.json
```

Expected: six runs; only the dance Base/DPO pair is ready at this gate.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_scail2_inference_plan.py tests/test_build_scail2_inference_plan.py manifests/inference_plan_sam2.json
git commit -m "Gate SCAIL2 inference on approved SAM2 masks"
```

---

### Task 9: Implement and Run the Dance Base/DPO Controller

**Files:**
- Create: `scripts/run_scail2_sam2_gate.py`
- Test: `tests/test_run_scail2_sam2_gate.py`
- Create after execution: `outputs/sam2_fallback_base_vs_dpo/mymotion__humandance__batch1_mingxing_cropped__002/{base.mp4,dpo.mp4,run_manifest.json}`

**Interfaces:**
- Consumes: Task 8 plan and Task 7 approved masks.
- Produces: sequential Base then DPO runs, isolated logs/exit files, ffprobe, hashes, and pair manifest.

- [ ] **Step 1: Write failing fail-closed tests**

```python
import unittest
from scripts.run_scail2_sam2_gate import select_runs

class ControllerTest(unittest.TestCase):
    def test_only_ready_pair_is_selected(self):
        plan = {"runs":[{"sample_id":"dance","variant":"base","ready":True},
                        {"sample_id":"dance","variant":"dpo","ready":True},
                        {"sample_id":"wide","variant":"base","ready":False}]}
        self.assertEqual([x["variant"] for x in select_runs(plan,"dance")], ["base","dpo"])
    def test_unready_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not ready"):
            select_runs({"runs":[{"sample_id":"dance","variant":"base","ready":False}]}, "dance")
```

- [ ] **Step 2: Verify failure**

Run: `/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_run_scail2_sam2_gate -v`

- [ ] **Step 3: Implement the controller**

```python
def select_runs(plan, sample_id):
    runs = [x for x in plan["runs"] if x["sample_id"] == sample_id]
    by_variant = {x["variant"]: x for x in runs}
    if set(by_variant) != {"base","dpo"} or not all(x["ready"] for x in runs):
        raise ValueError(f"sample pair not ready: {sample_id}")
    return [by_variant["base"], by_variant["dpo"]]
```

The CLI requires `--physical-gpu`, rejects `0`, sets `CUDA_VISIBLE_DEVICES`, runs stored Base then DPO commands with cwd equal to pinned WAN source, stops before DPO on Base failure, scans for Traceback/OOM/non-finite errors, ffprobes and hashes outputs, and records commands plus `SAM2 fallback masks`. It also asserts the two option maps differ only by save path and DPO LoRA path/alpha.

- [ ] **Step 4: Test, select GPU, and launch gate**

```bash
ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_run_scail2_sam2_gate -v
GPU=$(/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python scripts/select_idle_nonzero_gpu.py --minimum-free-mib 65000)
test "$GPU" != 0
tmux new-session -d -s scail2_sam2_base_dpo_gate_0808 \
  "cd '$ROOT' && /nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python scripts/run_scail2_sam2_gate.py --root '$ROOT' --plan '$ROOT/manifests/inference_plan_sam2.json' --sample-id mymotion__humandance__batch1_mingxing_cropped__002 --physical-gpu '$GPU' > logs/scail2_sam2_gate_controller.log 2>&1; printf '%s' \$? > logs/scail2_sam2_gate_controller.exit"
```

Monitor at low frequency. Do not change resolution, frames, sampling, CFG, or alpha after launch.

- [ ] **Step 5: Verify and commit compact evidence**

```bash
test "$(cat logs/scail2_sam2_gate_controller.exit)" = 0
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m json.tool outputs/sam2_fallback_base_vs_dpo/mymotion__humandance__batch1_mingxing_cropped__002/run_manifest.json >/dev/null
git add scripts/run_scail2_sam2_gate.py tests/test_run_scail2_sam2_gate.py outputs/sam2_fallback_base_vs_dpo/mymotion__humandance__batch1_mingxing_cropped__002/run_manifest.json
git commit -m "Record dance SCAIL2 SAM2 Base DPO gate"
```

Do not commit MP4 files or logs.

---

### Task 10: Validate Dance Outputs and Decide Expansion

**Files:**
- Create: `scripts/build_scail2_sam2_report.py`
- Test: `tests/test_build_scail2_sam2_report.py`
- Create after execution: `reports/sam2_fallback/dance_gate/report.json`
- Create after execution: `reports/sam2_fallback/dance_gate/comparison_5grid.mp4`
- Create after execution: `reports/sam2_fallback/dance_gate/frames/{frame_000.png,frame_048.png,frame_096.png}`

**Interfaces:**
- Consumes: source, target as evaluation-only, reference, masks, Base, DPO, and manifests.
- Produces: machine-readable QC and labeled comparison media.

- [ ] **Step 1: Write failing report tests**

Tests reject Base/DPO seed or condition-hash differences, reject a missing `SAM2 fallback masks` label, and accept a pair whose only extra DPO options are LoRA path/alpha.

- [ ] **Step 2: Verify failure**

Run: `/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_build_scail2_sam2_report -v`

- [ ] **Step 3: Implement report generation**

The report builder must verify condition/sampling equality; require 97 frames, 24 fps, and native dimensions; extract 0/48/96 from Source/Target/Base/DPO; build a five-view Source, Reference, Target (evaluation only), Base, DPO video; label generated views `SAM2 fallback masks`; record hashes and blank/decode checks; and provide review fields for identity, face, clothing/hair, motion, flicker, body edges, and background artifacts.

- [ ] **Step 4: Run and inspect**

```bash
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python scripts/build_scail2_sam2_report.py --root /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803 --sample-id mymotion__humandance__batch1_mingxing_cropped__002 --output reports/sam2_fallback/dance_gate
```

Inspect 0/48/96 and intervening motion frames. Expansion is allowed only when both videos decode, are nonblank, use the intended motion/conditions, and form a valid comparison. Poor DPO quality is a result; a broken chain is not.

- [ ] **Step 5: Commit the decision**

```bash
git add scripts/build_scail2_sam2_report.py tests/test_build_scail2_sam2_report.py reports/sam2_fallback/dance_gate/report.json
git commit -m "Audit dance SAM2 fallback Base DPO outputs"
```

If `conditioning_chain_valid=false`, stop and report evidence.

---

### Task 11: Expand the Frozen Pipeline to Two Remaining Samples

**Files:**
- Create after execution: `preprocessed_sam2/mymotion__051/`
- Create after execution: `preprocessed_sam2/mymotion__single_talking_face__datatrain__002/`
- Modify after execution: `manifests/inference_plan_sam2.json`
- Create after execution: remaining `run_manifest.json` and reports.

**Interfaces:**
- Consumes: unchanged pipeline and a valid dance report.
- Produces: masks and Base/DPO pairs at 1280x512 and 1280x704.

- [ ] **Step 1: Generate each mask bundle separately**

Select a fresh nonzero GPU for each sample, run `prepare_sam2_masks.py`, validate all 97 frames, inspect 0/48/96, and write QC review. Never reuse the dance box/mask.

- [ ] **Step 2: Rebuild and test plan**

```bash
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python scripts/build_scail2_inference_plan.py --root /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803 --output /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/manifests/inference_plan_sam2.json
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest tests.test_build_scail2_inference_plan -v
```

Expected: all six runs ready only after all three mask manifests pass.

- [ ] **Step 3: Run each pair**

Call the controller once for `mymotion__051` and once for `mymotion__single_talking_face__datatrain__002`. Parallel execution is permitted only if two nonzero GPUs independently exceed 65,000 MiB free; logs and outputs remain isolated.

- [ ] **Step 4: Build and inspect both reports**

Run the same report builder. Require valid ffprobe output and inspect 0/48/96 plus motion frames.

- [ ] **Step 5: Commit compact manifests**

```bash
git add manifests/inference_plan_sam2.json preprocessed_sam2/*/mask_manifest.json preprocessed_sam2/*/qc_review.json outputs/sam2_fallback_base_vs_dpo/*/run_manifest.json reports/sam2_fallback/*/report.json
git commit -m "Complete three sample SAM2 fallback comparison"
```

---

### Task 12: Final Audit and Local Result Package

**Files:**
- Create: `reports/sam2_fallback/final_audit.json`
- Create: `reports/sam2_fallback/README.md`
- Create locally: `D:\Code\SCAIL2_SAM2_fallback_Base_DPO_20260808\`

**Interfaces:**
- Consumes: successful manifests, reports, conditions, and outputs.
- Produces: compact local package and bounded research conclusion.

- [ ] **Step 1: Run complete verification**

```bash
cd /nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
/nvmedata/workspace2/users/nina/conda_envs/scail2_0803/bin/python -m unittest discover -s tests -p 'test_*.py' -v
git diff --check
git status --short
```

Expected: tests pass and only intended evidence is tracked.

- [ ] **Step 2: Build final audit**

Record input hashes, target-evaluation-only declaration, SAM2/YOLOX provenance, mask hashes, Base/DPO weight hashes, exact arguments, physical GPUs, ffprobe, output hashes, and disclaimer `non-official SAM2 mask preprocessing`.

- [ ] **Step 3: Copy compact results locally**

Copy design/plan, scripts/tests, manifests, QC overlays, comparison videos, Base/DPO MP4s, and concise logs to a new timestamped `D:\Code` directory. Exclude the 65 GB Base, SAM2 checkpoint, raw masks, source clones, environments, and wheels. Verify hashes after transfer.

- [ ] **Step 4: Write bounded conclusion**

For each resolution, state whether DPO changes identity, face detail, hair/clothing, flicker, body edges, motion, and background artifacts versus Base. State that target was evaluation-only and results are not equivalent to official SAM3 preprocessing.

- [ ] **Step 5: Commit final documentation**

```bash
git add reports/sam2_fallback/final_audit.json reports/sam2_fallback/README.md
git commit -m "Document SAM2 fallback Base DPO comparison"
```

Completion requires local hashes matching remote manifests and a conclusion that separates model behavior from mask-preprocessing limitations.
