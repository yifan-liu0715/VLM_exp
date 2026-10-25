from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
DWPOSE_COMMIT = "3dca5db79d9f9ffdd378753ddf6ec66535aace88"
SAM2_REPOSITORY = "https://github.com/facebookresearch/sam2.git"
DWPOSE_REPOSITORY = "https://github.com/IDEA-Research/DWPose.git"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)
YOLOX_URL = (
    "https://huggingface.co/yzd-v/DWPose/resolve/"
    "f7c16a3d45ad3783db41471848c80fbc281cabac/yolox_l.onnx?download=true"
)
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


def verify_asset(path: Path, spec: AssetSpec) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if spec.expected_sha256 and actual != spec.expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {spec.name}: expected "
            f"{spec.expected_sha256}, got {actual}"
        )
    return {
        **asdict(spec),
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": actual,
        "verified": True,
    }

