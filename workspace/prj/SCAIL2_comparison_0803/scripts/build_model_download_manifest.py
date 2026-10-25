from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path


ROOT = Path("/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803")
MODEL_ROOT = ROOT / "weights" / "SCAIL-2"
EXPECTED_DPO_SHA256 = (
    "cf29934c72fee6b24c3a5a82e8a9a4f3f4e93d76d988c9a20fac4e2b19e8fa09"
)

listing = json.loads((ROOT / "manifests" / "modelscope_files.json").read_text())
expected_sizes = {
    entry["Path"]: entry["Size"] for entry in listing if entry["Size"] > 0
}

actual_paths = sorted(
    path.relative_to(MODEL_ROOT).as_posix()
    for path in MODEL_ROOT.rglob("*")
    if path.is_file()
)
if set(actual_paths) != set(expected_sizes):
    raise RuntimeError(
        "Downloaded file set differs from official listing: "
        f"missing={sorted(set(expected_sizes) - set(actual_paths))}, "
        f"unexpected={sorted(set(actual_paths) - set(expected_sizes))}"
    )

hashes = {}
for line in (ROOT / "manifests" / "model_files.sha256").read_text().splitlines():
    digest, absolute_path = line.split(maxsplit=1)
    relative_path = Path(absolute_path).relative_to(MODEL_ROOT).as_posix()
    hashes[relative_path] = digest

if set(hashes) != set(actual_paths):
    raise RuntimeError("SHA256 manifest does not cover the complete model file set")

files = []
for relative_path in actual_paths:
    size = (MODEL_ROOT / relative_path).stat().st_size
    if size != expected_sizes[relative_path]:
        raise RuntimeError(
            f"Size mismatch for {relative_path}: {size} != {expected_sizes[relative_path]}"
        )
    files.append(
        {
            "path": relative_path,
            "size": size,
            "sha256": hashes[relative_path],
        }
    )

dpo_sha256 = hashes["model/bias-aware-dpo-lora.pt"]
if dpo_sha256 != EXPECTED_DPO_SHA256:
    raise RuntimeError(f"DPO SHA256 mismatch: {dpo_sha256}")

manifest = {
    "status": "passed",
    "source": "modelscope:ZhipuAI/SCAIL-2",
    "official_source_reference": "SCAIL-2 SAT README",
    "revision": "master",
    "modelscope_version": version("modelscope"),
    "verified_at": datetime.now(timezone.utc).isoformat(),
    "file_count": len(files),
    "total_bytes": sum(item["size"] for item in files),
    "expected_dpo_sha256": EXPECTED_DPO_SHA256,
    "actual_dpo_sha256": dpo_sha256,
    "files": files,
}

output = ROOT / "manifests" / "model_download.json"
output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: manifest[key] for key in (
    "status", "source", "revision", "file_count", "total_bytes", "actual_dpo_sha256"
)}, indent=2))
