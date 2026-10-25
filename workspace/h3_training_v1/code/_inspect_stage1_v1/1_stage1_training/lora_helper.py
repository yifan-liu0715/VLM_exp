#### lora utils adapted from diffsynth ####
import torch
from diffsynth.core import load_state_dict
from peft import LoraConfig, inject_adapter_in_model

def add_lora_to_model(
    model,
    lora_rank=128,
    alpha=1, # default 1 (lora_rank=lora_alpha)
    pretrained_lora_path=None,
    lora_target_modules="q,k,v,o,ffn.0,ffn.2",
):

    if alpha == 1:
        lora_alpha = lora_rank
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=lora_target_modules.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    
    if pretrained_lora_path is not None:
        print(f"Loading pretrained LoRA from {pretrained_lora_path}...")
        lora = load_state_dict(pretrained_lora_path)
        def fix_lora_state_dict_keys(lora_state_dict):
            fixed = {}
            for k, v in lora_state_dict.items():
                if not k.startswith("model."):
                    fixed["model." + k] = v
                else:
                    fixed[k] = v
            return fixed
        # lora = fix_lora_state_dict_keys(lora)
        missing_keys, unexpected_keys = model.load_state_dict(lora, strict=False)
        print(f"Loaded pretrained LoRA. Unexpected keys: {unexpected_keys}")
    return model

def clear_lora(model, verbose=1):
    cleared_num = 0
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.data.zero_()
            cleared_num += 1
    if verbose >= 1:
        print(f"{cleared_num} LoRA parameters are zeroed.")
    return model

def get_name_dict(lora_state_dict):
    lora_name_dict = {}
    for key in lora_state_dict:
        if ".lora_up." in key:
            lora_A_key = "lora_down"
            lora_B_key = "lora_up"
        else:
            lora_A_key = "lora_A"
            lora_B_key = "lora_B"
        if lora_B_key not in key:
            continue
        keys = key.split(".")
        if len(keys) > keys.index(lora_B_key) + 2:
            keys.pop(keys.index(lora_B_key) + 1)
        keys.pop(keys.index(lora_B_key))
        if keys[0] == "diffusion_model":
            keys.pop(0)
        keys.pop(-1)
        target_name = ".".join(keys)
        lora_name_dict[target_name] = (key, key.replace(lora_B_key, lora_A_key))
    return lora_name_dict

def convert_state_dict(state_dict, suffix=".weight"):
    name_dict = get_name_dict(state_dict)
    state_dict_ = {}
    for name in name_dict:
        weight_up = state_dict[name_dict[name][0]]
        weight_down = state_dict[name_dict[name][1]]
        state_dict_[name + f".lora_B{suffix}"] = weight_up
        state_dict_[name + f".lora_A{suffix}"] = weight_down
    return state_dict_
    
def fuse_lora_to_base_model(model: torch.nn.Module, state_dict, alpha=1.0):
    updated_num = 0
    state_dict = convert_state_dict(state_dict)
    lora_layer_names = set([i.replace(".lora_B.weight", "") for i in state_dict if i.endswith(".lora_B.weight")])
    for name, module in model.named_modules():
        if name in lora_layer_names:
            param = next(module.parameters())
            weight_up = state_dict[name + ".lora_B.weight"].to(device=param.device, dtype=param.dtype)
            weight_down = state_dict[name + ".lora_A.weight"].to(device=param.device, dtype=param.dtype)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)
            state_dict_base = module.state_dict()
            state_dict_base["weight"] = state_dict_base["weight"].to(device=param.device, dtype=param.dtype) + weight_lora
            module.load_state_dict(state_dict_base)
            updated_num += 1
    print(f"{updated_num} tensors are fused by LoRA. Fused LoRA layers cannot be cleared by `clear_lora()`.")