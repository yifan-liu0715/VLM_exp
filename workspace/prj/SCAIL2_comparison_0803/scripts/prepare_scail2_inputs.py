#!/usr/bin/env python3
"""Prepare immutable SCAIL-2 comparison inputs from Scheme4 run manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SAMPLE_IDS = (
    "mymotion__051",
    "mymotion__single_talking_face__datatrain__002",
    "mymotion__humandance__batch1_mingxing_cropped__002",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mapping(data: dict, sample_id: str) -> None:
    if data.get("sample_id") != sample_id:
        raise ValueError(f"sample mismatch: expected {sample_id}, got {data.get('sample_id')}")
    expected_segments = {
        "source_video": "/output_video/",
        "target_video": "/input_video/",
        "reference_image": "/reference/",
    }
    for key, segment in expected_segments.items():
        value = data.get(key)
        if not isinstance(value, str) or segment not in value.replace("\\", "/"):
            raise ValueError(f"{sample_id}: {key} must contain {segment}, got {value!r}")
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    if Path(data["source_video"]).resolve() == Path(data["target_video"]).resolve():
        raise ValueError(f"{sample_id}: source and target resolve to the same file")


def replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {link}")
    os.symlink(target, link)


def prepare(manifest_root: Path, output_root: Path) -> dict:
    records = []
    for sample_id in SAMPLE_IDS:
        source_manifest = manifest_root / sample_id / "run_manifest.json"
        if not source_manifest.is_file():
            raise FileNotFoundError(source_manifest)
        with source_manifest.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        validate_mapping(data, sample_id)

        source = Path(data["source_video"])
        target = Path(data["target_video"])
        reference = Path(data["reference_image"])
        sample_dir = output_root / sample_id
        links = {
            "driving.mp4": source,
            "target.mp4": target,
            "ref.jpg": reference,
        }
        for name, original in links.items():
            replace_symlink(sample_dir / name, original)

        (sample_dir / "prompt.txt").write_text(str(data["prompt"]).strip() + "\n", encoding="utf-8")
        records.append(
            {
                "sample_id": sample_id,
                "split": data.get("split"),
                "prompt": data["prompt"],
                "scheme4_run_manifest": str(source_manifest),
                "scheme4_run_manifest_sha256": sha256(source_manifest),
                "source_video": str(source),
                "source_sha256": sha256(source),
                "target_video": str(target),
                "target_sha256": sha256(target),
                "reference_image": str(reference),
                "reference_sha256": sha256(reference),
                "input_dir": str(sample_dir),
                "scail2_files": {name: str(sample_dir / name) for name in links},
                "width": data.get("width"),
                "height": data.get("height"),
                "num_frames": data.get("num_frames"),
                "fps": data.get("fps"),
            }
        )
    return {
        "status": "passed",
        "mapping": {
            "SCAIL-2 driving.mp4": "Scheme4 source_video (dataset output_video)",
            "comparison target.mp4": "Scheme4 target_video (dataset input_video)",
            "SCAIL-2 ref.jpg": "Scheme4 reference_image (dataset reference)",
        },
        "uses_symlinks": True,
        "samples": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.manifest_root, args.output_root)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "samples": len(result["samples"])}, sort_keys=True))


if __name__ == "__main__":
    main()
