from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


SOURCE = Path("/tmp/decord_wheel_audit/decord-0.6.0-py3-none-manylinux2010_x86_64.whl")
UNPACK_ROOT = Path("/tmp/decord_wheel_unpack_manual")
PACKAGE_ROOT = UNPACK_ROOT / "decord-0.6.0"
OUTPUT_ROOT = Path("/tmp/decord_wheel_repaired")

shutil.rmtree(UNPACK_ROOT, ignore_errors=True)
shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
PACKAGE_ROOT.mkdir(parents=True)
OUTPUT_ROOT.mkdir(parents=True)

with zipfile.ZipFile(SOURCE) as archive:
    archive.extractall(PACKAGE_ROOT)

wheel_metadata = PACKAGE_ROOT / "decord-0.6.0.dist-info" / "WHEEL"
original = wheel_metadata.read_text()
repaired = original.replace(
    "Tag: cp36-cp36m-manylinux2010_x86_64",
    "Tag: py3-none-manylinux2010_x86_64",
)
if repaired == original:
    raise RuntimeError("Expected broken decord wheel tag was not found")
wheel_metadata.write_text(repaired)

subprocess.run(
    [
        sys.executable,
        "-m",
        "wheel",
        "pack",
        str(PACKAGE_ROOT),
        "--dest-dir",
        str(OUTPUT_ROOT),
        "--build-number",
        "1",
    ],
    check=True,
)

outputs = list(OUTPUT_ROOT.glob("*.whl"))
if len(outputs) != 1:
    raise RuntimeError(f"Expected one repaired wheel, found {outputs}")
print(outputs[0])
