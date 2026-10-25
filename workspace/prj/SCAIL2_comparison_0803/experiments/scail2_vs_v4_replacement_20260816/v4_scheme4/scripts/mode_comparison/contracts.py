"""Frozen, model-independent contracts for the two comparison modes."""

from __future__ import annotations

from dataclasses import dataclass


REMOTE_ROOT = "/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803"
DANCE_SAMPLE_ID = "mymotion__humandance__batch1_mingxing_cropped__002"
SOURCE_VIDEO = (
    "/nvmedata/workspace2/users/nina/dataset_video_train/704x1280/"
    f"output_video/{DANCE_SAMPLE_ID}.mp4"
)
TARGET_VIDEO = (
    "/nvmedata/workspace2/users/nina/dataset_video_train/704x1280/"
    f"input_video/{DANCE_SAMPLE_ID}.mp4"
)
REFERENCE_IMAGE = (
    "/nvmedata/workspace2/users/nina/dataset_video_train/704x1280/"
    f"reference/{DANCE_SAMPLE_ID}.jpg"
)
PROMPT = (
    "A slim young woman with long dark hair, wearing a black and white maid "
    "outfit and cat ears, performs a rhythmic dance. She starts with her back "
    "to the camera, turns fully around, and sways her hips while moving her "
    "arms. The setting is a minimalist room with white walls, a wooden chair, "
    "and a small framed picture."
)

SCAIL2_WEIGHT = f"{REMOTE_ROOT}/weights/converted/SCAIL-2.safetensors"
SCAIL2_SHA256 = "8b844f026510f0b94d3209d66444b403a089ec179dc149d85513c13b2fa37118"
LATEST_LORA_WEIGHT = (
    "/nvmedata/workspace2/users/nina/prj/DiffSynth-Studio/"
    "v2v_reference_independent_posenc_scheme4_0729/recommended/final_adapter/"
    "v2v_lora_source_reference_posenc.safetensors"
)
LATEST_LORA_SHA256 = "dece6fe3aa1de90166f861c4e10405ab32ceb4b1a78f61e9bac08af5f595908a"
MASK_PREPROCESSOR = "SAM2 fallback reproducing SCAIL replacement palette; not official SAM3"

CANVAS_MAP = {
    (1280, 512): (960, 384),
    (1280, 704): (928, 512),
    (704, 1280): (512, 928),
}


@dataclass(frozen=True)
class ExperimentContract:
    mode: str
    sample_id: str = DANCE_SAMPLE_ID
    width: int = 512
    height: int = 928
    input_frames: int = 97
    input_fps: int = 24
    output_frames: int = 65
    output_fps: int = 16
    solver: str = "unipc"
    steps: int = 40
    guidance_scale: float = 5.0
    shift: float = 3.0
    seed: int = 42
    segment_len: int = 65
    segment_overlap: int = 5
    scail2_weight: str = SCAIL2_WEIGHT
    scail2_sha256: str = SCAIL2_SHA256
    lora_weight: str = LATEST_LORA_WEIGHT
    lora_sha256: str = LATEST_LORA_SHA256

    @property
    def experiment_name(self) -> str:
        return f"{self.mode}_vs_lora_20260812"

    @property
    def experiment_root(self) -> str:
        return f"{REMOTE_ROOT}/experiments/{self.experiment_name}"


def temporal_indices(input_frames: int, output_frames: int) -> tuple[int, ...]:
    """Select ordered frames across both endpoints using round-half-up."""
    input_frames = int(input_frames)
    output_frames = int(output_frames)
    if input_frames < 2 or output_frames < 2:
        raise ValueError("input_frames and output_frames must both be at least 2")
    if output_frames > input_frames:
        raise ValueError("output_frames cannot exceed input_frames")
    denominator = output_frames - 1
    return tuple(
        (2 * index * (input_frames - 1) + denominator) // (2 * denominator)
        for index in range(output_frames)
    )


def canvas_for(width: int, height: int) -> tuple[int, int]:
    try:
        return CANVAS_MAP[(int(width), int(height))]
    except KeyError as error:
        raise ValueError(f"unsupported source canvas: {width}x{height}") from error


def comparison_contract(mode: str) -> ExperimentContract:
    normalized = str(mode).strip().lower()
    if normalized not in {"replacement", "animation"}:
        raise ValueError(f"unsupported SCAIL-2 mode: {mode}")
    return ExperimentContract(mode=normalized)
