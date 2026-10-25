from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


REPO = Path("/nvmedata/workspace2/users/nina/prj/SCAIL2_comparison_0803/code/SCAIL-2")
sys.path.insert(0, str(REPO))

MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "deepspeed": "deepspeed",
    "av": "av",
    "decord": "decord",
    "sat": None,
    "sgm": None,
}

for module_name, distribution_name in MODULES.items():
    module = import_module(module_name)
    if distribution_name is None:
        module_version = "source-tree"
    else:
        try:
            module_version = version(distribution_name)
        except PackageNotFoundError:
            module_version = getattr(module, "__version__", "unknown")
    print(f"IMPORT_OK {module_name} {module_version} {getattr(module, '__file__', None)}")

print("ALL_IMPORTS_OK")
