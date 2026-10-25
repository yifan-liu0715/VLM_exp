import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = self._gather_state_dict(model)
        if accelerator.is_main_process:
            state_dict = self.state_dict_converter(state_dict)
            state_dict = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state_dict.items()}
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = self._gather_state_dict(model)
        if accelerator.is_main_process:
            if len(state_dict) == 0:
                raise RuntimeError(
                    f"Refusing to write an empty checkpoint to {file_name}: no trainable "
                    f"parameters were gathered. Check the key cleaning in _gather_fsdp."
                )
            state_dict = self.state_dict_converter(state_dict)
            state_dict = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state_dict.items()}
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


    def _gather_state_dict(self, model):
        """Gather trainable parameters from DeepSpeed ZeRO-3 or FSDP wrapped model."""
        dit = model.pipe.dit

        # DeepSpeed engine
        if hasattr(dit, 'module') and hasattr(dit, 'zero_optimization_stage'):
            return self._gather_deepspeed(dit)

        # FSDP wrapped
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(dit, FSDP):
            return self._gather_fsdp(dit)

        # Plain model (no sharding)
        state_dict = {}
        for name, param in dit.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.clone().cpu()
        return state_dict


    def _gather_deepspeed(self, ds_engine):
        """Gather from DeepSpeed ZeRO-3 partitioned model."""
        import deepspeed
        trainable_names = {n for n, p in ds_engine.module.named_parameters() if p.requires_grad}
        trainable_params = [p for n, p in ds_engine.module.named_parameters() if p.requires_grad]
        with deepspeed.zero.GatheredParameters(trainable_params):
            state_dict = {}
            for name, param in ds_engine.module.named_parameters():
                if name in trainable_names:
                    state_dict[name] = param.data.clone().cpu()
        return state_dict


    def _gather_fsdp(self, fsdp_model):
        """Gather from FSDP sharded model."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            full_state_dict = fsdp_model.state_dict()

        # FULL_STATE_DICT strips the FSDP wrapper prefixes from its keys but
        # named_parameters() keeps them, so the two sets only intersect once both
        # are cleaned. Without this every checkpoint comes out empty.
        def clean(name):
            return name.replace("_fsdp_wrapped_module.", "")

        trainable_names = {
            clean(name) for name, param in fsdp_model.named_parameters() if param.requires_grad
        }

        state_dict = {}
        for name, tensor in full_state_dict.items():
            name = clean(name)
            if name in trainable_names:
                state_dict[name] = tensor
        return state_dict
