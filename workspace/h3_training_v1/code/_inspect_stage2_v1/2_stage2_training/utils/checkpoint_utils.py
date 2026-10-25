"""
独立的检查点保存工具，不依赖 Lightning 框架
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


def save_fsdp_checkpoint_with_topk(
    model,
    save_dir,
    global_step,
    only_trainable=True,
    top_k=50,
    saved_checkpoints_list=None,
    prefix="model"
):
    """
    保存 FSDP 模型检查点并管理 top_k
    
    Args:
        model: FSDP wrapped model
        save_dir: 保存目录
        global_step: 当前训练步数
        only_trainable: 是否只保存可训练参数
        top_k: 保留的最大检查点数量
        saved_checkpoints_list: 已保存检查点的列表 [(step, path), ...]
        prefix: 文件名前缀
    
    Returns:
        updated saved_checkpoints_list
    """
    if saved_checkpoints_list is None:
        saved_checkpoints_list = []
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用 FSDP 的 state_dict API 收集模型状态
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        consolidated_model_state_dict = model.state_dict()
        
        if only_trainable:
            # 获取可训练参数的名称（去掉 FSDP 包装的前缀）
            trainable_param_names = {
                name.replace('_fsdp_wrapped_module.', '') 
                for name, param in model.named_parameters() 
                if param.requires_grad
            }
            # 过滤出可训练参数
            filtered_state_dict = {
                name: tensor 
                for name, tensor in consolidated_model_state_dict.items() 
                if name in trainable_param_names
            }
        else:
            filtered_state_dict = consolidated_model_state_dict
        
        # 只有 rank 0 保存检查点
        is_rank_0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if is_rank_0:
            ckpt_path = os.path.join(save_dir, f"step_{global_step}_{prefix}.pt")
            torch.save(filtered_state_dict, ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")

            # 管理保存的文件，仅保留 top_k 个
            saved_checkpoints_list.append((global_step, ckpt_path))
            saved_checkpoints_list.sort(reverse=True)  # 最新的在前

            if len(saved_checkpoints_list) > top_k:
                # 删除最旧的
                _, old_path = saved_checkpoints_list.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"[Checkpoint] Removed old: {old_path}")
        
        # 清理内存
        del consolidated_model_state_dict
        if only_trainable:
            del filtered_state_dict
    
    # 同步所有进程
    if dist.is_initialized():
        dist.barrier()
    
    return saved_checkpoints_list



# ---------------------------------------------------------------------------
# Resume support. Upstream stage2 is single-shot: `self.step = 0` is hardcoded,
# `ckpt_step` in default_config.yaml is never read, and nothing is ever loaded
# back. Under a 2 h Slurm wall that caps the run at ~200 steps, because every
# array task restarts from zero and redoes the same segment.
#
# What has to be restored, and where it lives:
#   generator / critic LoRA -> step_N_{generator,critic}_lora.pt (already saved)
#   AdamW moments           -> step_N_{generator,critic}_optim.pt (added here)
#   step counter            -> derived from the filename
#   pose_patch_embedding    -> NOT here. It is frozen, so only_trainable=True
#                              filters it out; it comes from the stage1
#                              checkpoint via lora_ckpt at construction time.
# ---------------------------------------------------------------------------

def find_latest_resume_step(save_dir, prefixes=("generator_lora", "critic_lora")):
    """Highest step for which every required file is present.

    Requiring all of them matters: a wall-clock kill mid-save can leave the
    generator written and the critic not, and resuming off that pair would
    silently pair mismatched weights.
    """
    import re
    if not os.path.isdir(save_dir):
        return None
    steps = {}
    for fn in os.listdir(save_dir):
        m = re.match(r"step_(\d+)_(.+)\.pt$", fn)
        if m:
            steps.setdefault(int(m.group(1)), set()).add(m.group(2))
    complete = [s for s, got in steps.items() if all(p in got for p in prefixes)]
    return max(complete) if complete else None


def save_optimizer_state(model, optimizer, save_dir, global_step, prefix):
    """Consolidated AdamW state next to the weights, written atomically."""
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        osd = FSDP.optim_state_dict(model, optimizer)
    is_rank_0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if is_rank_0:
        path = os.path.join(save_dir, f"step_{global_step}_{prefix}_optim.pt")
        tmp = path + ".tmp"
        torch.save({"step": global_step, "optimizer": osd}, tmp)
        os.replace(tmp, path)
        print(f"[Checkpoint] Saved: {path}")


def load_lora_state(model, path):
    """Load a trainable-only state dict back into an FSDP-wrapped model."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
    ):
        missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"{path}: {len(unexpected)} unexpected keys, e.g. {list(unexpected)[:4]}")
    matched = len(sd) - len(unexpected)
    if matched == 0:
        raise RuntimeError(f"{path} matched no parameter; resume would silently do nothing.")
    print(f"[Resume] {os.path.basename(path)}: {matched}/{len(sd)} tensors restored")


def load_optimizer_state(model, optimizer, path, expected_step):
    """Restore AdamW moments, refusing state that belongs to another step."""
    if not os.path.exists(path):
        print(f"[Resume] no optimizer state at {path}; moments restart from zero")
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("step") != expected_step:
        print(f"[Resume] {path} is at step {payload.get('step')} but weights are at "
              f"{expected_step}; ignoring it rather than pairing mismatched state")
        return
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
        FullOptimStateDictConfig(rank0_only=False, offload_to_cpu=True),
    ):
        osd = FSDP.optim_state_dict_to_load(model, optimizer, payload["optimizer"])
        optimizer.load_state_dict(osd)
    print(f"[Resume] optimizer moments restored from step {expected_step}")
