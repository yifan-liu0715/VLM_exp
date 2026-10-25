#!/usr/bin/env python3
"""Fetch and verify the official assets used by the SAM2 fallback path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam2_fallback_0808.assets import (  # noqa: E402
    DWPOSE_COMMIT,
    DWPOSE_REPOSITORY,
    SAM2_CHECKPOINT_URL,
    SAM2_COMMIT,
    SAM2_REPOSITORY,
    YOLOX_SHA256,
    YOLOX_URL,
    AssetSpec,
    verify_asset,
)


def run_checked(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def ensure_repository(path: Path, url: str, commit: str, verify_only: bool) -> dict:
    if path.is_dir():
        actual = run_checked(["git", "-C", str(path), "rev-parse", "HEAD"])
        if actual == commit:
            return {
                "path": str(path.resolve()),
                "repository": url,
                "commit": actual,
                "verified": True,
            }
        if verify_only:
            raise ValueError(f"source commit mismatch for {path}: {actual}")
    else:
        if verify_only:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        run_checked(["git", "clone", "--filter=blob:none", url, str(path)])
    if not verify_only:
        run_checked(["git", "-C", str(path), "fetch", "--depth", "1", "origin", commit])
        run_checked(["git", "-C", str(path), "checkout", "--detach", commit])
    actual = run_checked(["git", "-C", str(path), "rev-parse", "HEAD"])
    if actual != commit:
        raise ValueError(f"source commit mismatch for {path}: {actual}")
    return {"path": str(path.resolve()), "repository": url, "commit": actual, "verified": True}


def download_atomic(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "scail2-sam2-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(path)


def load_existing_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def fetch_or_verify(spec: AssetSpec, verify_only: bool) -> dict:
    path = Path(spec.path)
    if not path.is_file():
        if verify_only:
            raise FileNotFoundError(path)
        download_atomic(spec.url, path)
    return verify_asset(path, spec)


def build_manifest(root: Path, verify_only: bool) -> dict:
    manifest_path = root / "manifests" / "sam2_fallback_assets.json"
    sam2_source = root / "code" / f"sam2-{SAM2_COMMIT}"
    dwpose_source = root / "code" / f"DWPose-{DWPOSE_COMMIT}"
    weights = root / "weights" / "sam2_fallback"
    sam2_checkpoint = weights / "sam2.1_hiera_large.pt"
    yolox_checkpoint = weights / "yolox_l.onnx"
    existing = load_existing_manifest(manifest_path)

    sources = {
        "sam2": ensure_repository(sam2_source, SAM2_REPOSITORY, SAM2_COMMIT, verify_only),
        "dwpose": ensure_repository(dwpose_source, DWPOSE_REPOSITORY, DWPOSE_COMMIT, verify_only),
    }
    recorded_sam2_hash = existing.get("checkpoints", {}).get("sam2", {}).get("sha256")
    checkpoints = {
        "sam2": fetch_or_verify(
            AssetSpec(
                "sam2.1_hiera_large",
                SAM2_CHECKPOINT_URL,
                str(sam2_checkpoint),
                recorded_sam2_hash,
            ),
            verify_only,
        ),
        "yolox": fetch_or_verify(
            AssetSpec("yolox_l", YOLOX_URL, str(yolox_checkpoint), YOLOX_SHA256),
            verify_only,
        ),
    }
    return {
        "manifest_version": 1,
        "status": "verified",
        "generated_at": existing.get("generated_at", datetime.now(timezone.utc).isoformat()),
        "sources": sources,
        "checkpoints": checkpoints,
    }


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    data = build_manifest(root, args.verify_only)
    write_manifest(root / "manifests" / "sam2_fallback_assets.json", data)
    print(json.dumps({"status": data["status"], "verify_only": args.verify_only}))


if __name__ == "__main__":
    main()
