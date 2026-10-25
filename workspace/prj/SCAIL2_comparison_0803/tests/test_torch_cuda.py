import json

import torch
import torchvision


assert torch.__version__ == "2.9.1+cu126", torch.__version__
assert torchvision.__version__ == "0.24.1+cu126", torchvision.__version__
assert torch.cuda.is_available()

x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
y = x @ x.T
torch.cuda.synchronize()

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "visible_device_count": torch.cuda.device_count(),
            "visible_device_name": torch.cuda.get_device_name(0),
            "result_sum": float(y.sum()),
        },
        sort_keys=True,
    )
)
