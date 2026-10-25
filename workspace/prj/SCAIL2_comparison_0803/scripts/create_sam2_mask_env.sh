#!/usr/bin/env bash
set -euo pipefail

ROOT=/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803
SOURCE_ENV=/nvmedata/workspace2/users/nina/conda_envs/scail2_0803
ENV=/nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808
SAM2_SOURCE="$ROOT/code/sam2-2b90b9f5ceec907a1c18123530e92e794ad901a4"

available_kb=$(df --output=avail /nvmedata | tail -n 1)
if [ "$available_kb" -lt $((30 * 1024 * 1024)) ]; then
  echo "Refusing install: /nvmedata has less than 30 GiB free" >&2
  exit 20
fi

if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" --clone "$SOURCE_ENV"
fi

repair_pip_clone_if_needed() {
  local site_packages="$ENV/lib/python3.11/site-packages"
  local dist_info_count
  dist_info_count=$(find "$site_packages" -maxdepth 1 -type d -name 'pip-*.dist-info' | wc -l)

  if "$ENV/bin/python" -m pip --version >/dev/null 2>&1 && [ "$dist_info_count" -eq 1 ]; then
    return
  fi

  # `conda create --clone` can copy a pip-upgraded package tree and then
  # install the older conda-recorded pip over it. Rebuild pip from one
  # package source before invoking pip itself.
  case "$ENV" in
    /nvmedata/workspace2/users/nina/conda_envs/sam2_mask_0808) ;;
    *) echo "Refusing pip repair outside the isolated SAM2 environment" >&2; exit 21 ;;
  esac
  rm -rf -- "$site_packages/pip" "$site_packages"/pip-*.dist-info
  conda install -y -p "$ENV" --force-reinstall pip=26.1.2
  "$ENV/bin/python" -m pip --version
}

repair_pip_clone_if_needed

test -d "$SAM2_SOURCE"
"$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV/bin/python" -m pip install onnxruntime-gpu hydra-core iopath tqdm
"$ENV/bin/python" -m pip install --no-build-isolation -e "$SAM2_SOURCE"
"$ENV/bin/python" -m pip check

conda list -p "$ENV" --explicit > "$ROOT/manifests/sam2_mask_conda_explicit.txt"
"$ENV/bin/python" -m pip freeze > "$ROOT/manifests/sam2_mask_pip_freeze.txt"
