"""
train_fsdp_profile.py — DeepSpeed ZeRO-3 多节点训练（torchrun 启动）

用法：
  torchrun --nnodes=2 --nproc_per_node=8 \
    --master_addr=$MASTER_IP --master_port=29500 --node_rank=$MACHINE_RANK \
    train_fsdp_profile.py [args...]
"""

import sys, os, json, math, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import torch
import argparse
import accelerate
import warnings
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
from functools import partial

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.data_v2v import AnimateVideoDataset, TokenBucketSampler
from diffsynth.core.data.data_render import RenderVideoDataset
from diffsynth.pipelines.wan_video_causal_animate_training_motion import WanVideoPipeline
from diffsynth.diffusion import *
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.utils.data import save_video, VideoData

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────────────────────
# Training module
# ─────────────────────────────────────────────────────────────

class WanFullParamTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths,
        tokenizer_path,
        device="cpu",
        task="sft:causal_incontext_tsf",
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        tf_clean_aug_max_timestep=0,
        tf_sync_noise_to_clean=0,
        tsf_ref_noise_timestep_id=None,
        ref_noise_start_block: int = 2,
        use_gradient_checkpointing=True,
        bid_mode=False,
        extra_trainable_modules=None,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Forcing gradient checkpointing on to avoid OOM.")
            use_gradient_checkpointing = True

        print(f"[rank {os.environ.get('RANK', 0)}] Loading pipeline on device={device}")
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_paths=model_paths,
            tokenizer_path=tokenizer_path,
        )

        self.pipe.dit.num_frame_per_block = 3

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        self.pipe.scheduler.set_timesteps(1000, training=True)
        import copy
        self.pipe.infer_scheduler = copy.deepcopy(self.pipe.scheduler)
        if lora_base_model is None:
            frozen_top = ("vae", "text_encoder", "image_encoder", "tokenizer")
            frozen_dit_submodules = ("pose_patch_embedding", "motion_encoder", "face_adapter", "face_encoder")
            for name, model in self.pipe.named_children():
                if name in frozen_top:
                    for p in model.parameters():
                        p.requires_grad_(False)
                elif name == "dit":
                    for p in model.parameters():
                        p.requires_grad_(True)
                    for sub_name in frozen_dit_submodules:
                        sub = getattr(model, sub_name, None)
                        if sub is not None:
                            for p in sub.parameters():
                                p.requires_grad_(False)
        else:
            self.switch_pipe_to_training_mode(
                self.pipe,
                trainable_models=trainable_models,
                lora_base_model=lora_base_model,
                lora_target_modules=lora_target_modules,
                lora_rank=lora_rank,
                lora_checkpoint=lora_checkpoint,
                preset_lora_path=preset_lora_path,
                preset_lora_model=preset_lora_model,
                task=task,
            )

        # LoRA only covers the attention/ffn projections, so any DiT submodule that
        # the base checkpoint did not provide would stay frozen at its init.
        for name in (extra_trainable_modules or "").split(","):
            name = name.strip()
            if not name:
                continue
            sub = getattr(self.pipe.dit, name, None)
            if sub is None:
                raise ValueError(f"--extra_trainable_modules: dit has no submodule {name!r}")
            for p in sub.parameters():
                p.requires_grad_(True)
            print(f"[train] unfroze dit.{name}")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.task = task
        self.task_to_loss = {
            "sft": lambda pipe, s, p, n: FlowMatchSFTLoss(pipe, **s, **p),
            "sft:causal": lambda pipe, s, p, n: FlowMatchSFTLoss(pipe, **s, **p),
            "sft:causal_incontext": lambda pipe, s, p, n: FlowMatchInContextSFTLoss(pipe, **s, **p),
            "sft:causal_incontext_df": lambda pipe, s, p, n: FlowMatchInContextDiffusionForcingLoss(pipe, **s, **p),
            "sft:causal_incontext_tf": lambda pipe, s, p, n: FlowMatchInContextTeacherForcingLoss(pipe, **s, **p),
            "sft:causal_incontext_tsf": lambda pipe, s, p, n: FlowMatchInContextTimeStepForcingLoss(pipe, **s, **p),
            "sft:bid_incontext_tsf": lambda pipe, s, p, n: FlowMatchInContextTimeStepForcingLoss(pipe, **s, **p),
            "sft:causal_incontext_tsf_refnoise": lambda pipe, s, p, n: FlowMatchInContextTimeStepForcingLossRefNoiseSchedule(pipe, **s, **p),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.tf_clean_aug_max_timestep = tf_clean_aug_max_timestep
        self.tf_sync_noise_to_clean = tf_sync_noise_to_clean
        self.tsf_ref_noise_timestep_id = tsf_ref_noise_timestep_id
        self.ref_noise_start_block = ref_noise_start_block
        self.bid_mode = bid_mode

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["text"]}
        inputs_nega = {}
        inputs_shared = {
            "input_image": data["ref_image"].squeeze(0),
            "input_video": data["tgt_frame"],
            "motion_video": data["motion_frame"],
            "height": data["tgt_frame"].shape[-2],
            "width": data["tgt_frame"].shape[-1],
            "num_frames": data["tgt_frame"].shape[1],
            "animate_pose_video": data["tgt_frame_pose"],
            "animate_face_video": data.get("tgt_frame_face", None),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": False,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "tf_clean_aug_max_timestep": self.tf_clean_aug_max_timestep,
            "tf_sync_noise_to_clean": self.tf_sync_noise_to_clean,
            "tsf_ref_noise_timestep_id": self.tsf_ref_noise_timestep_id,
            "ref_noise_start_block": self.ref_noise_start_block,
            "bid_mode": self.bid_mode
        }
        return inputs_shared, inputs_posi, inputs_nega

    def _encode_prompt(self, prompt):
        pipe = self.pipe
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def forward(self, data, vis=False, clean_x=None):
        if vis:
            return self.forward_vis(data, clean_x=clean_x)
        return self.forward_train(data)

    def forward_train(self, data):
        device = self.pipe.device
        inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, device, self.pipe.torch_dtype)
        inputs_shared, inputs_posi, inputs_nega = inputs
        if "prompt" in inputs_posi and inputs_posi["prompt"] is not None:
            inputs_posi["context"] = self._encode_prompt(inputs_posi["prompt"])
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss

    def forward_vis(self, data, num_steps=10, clean_x=None):
        pipe = self.pipe
        inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
        inputs_shared, inputs_posi, inputs_nega = inputs
        if "prompt" in inputs_posi and inputs_posi["prompt"] is not None:
            inputs_posi["context"] = self._encode_prompt(inputs_posi["prompt"])
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in pipe.units:
            inputs = pipe.unit_runner(unit, pipe, *inputs)
        inputs_shared, inputs_posi, inputs_nega = inputs

        latents = inputs_shared.get("latents")
        if latents is None:
            return None

        if clean_x is not None and clean_x is not False:
            cond_lens = int(inputs_shared.get("cond_lens", 1))
            input_latents = inputs_shared.get("input_latents")
            if input_latents is not None:
                cond_frames = input_latents[:, :, :cond_lens]
                content_frames = input_latents[:, :, cond_lens:]
                clean_x = torch.cat([cond_frames, content_frames], dim=2)
            else:
                clean_x = None

        pipe.infer_scheduler.set_timesteps(num_steps)
        ref_latents = latents[:, :, 0:1]
        content_latents = latents[:, :, 1:]
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        B, _, T_content, _, _ = content_latents.shape
        for progress_id, ts in enumerate(pipe.infer_scheduler.timesteps):
            ts_t = ts.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared["latents"] = torch.cat([ref_latents, content_latents], dim=2)
            if clean_x is not None:
                ts_t_2d = ts_t.unsqueeze(0).expand(B, T_content)  # (B, T) for TF mode: clean tokens get zeros t_mod
                pred = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=ts_t_2d, clean_x=clean_x)
            else:
                pred = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=ts_t)
            content_latents = pipe.infer_scheduler.step(pred, pipe.infer_scheduler.timesteps[progress_id], content_latents)

        return torch.cat([ref_latents, content_latents], dim=2)


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_vis(accelerator, model, data, vis_dir, step, clean_x=None):
    rank = accelerator.process_index
    os.makedirs(vis_dir, exist_ok=True)

    pipe = model.pipe
    sample = {k: (v[:1] if isinstance(v, torch.Tensor) else ([v[0]] if isinstance(v, list) else v))
              for k, v in data.items()}

    # The denoising loop runs collectives inside FSDP. Swallowing an exception
    # around it would drop this rank out of the remaining all-gathers while the
    # other seven block on them until the NCCL watchdog fires -- the same job
    # loss, ten minutes later. So let it propagate and end the job promptly; the
    # array restarts from the last checkpoint. Release the training allocations
    # first, since OOM on top of them is the likeliest way this fails.
    torch.cuda.empty_cache()
    latents = model(sample, vis=True, clean_x=clean_x)

    # Decoding and writing touch no collectives (the VAE is not FSDP-wrapped),
    # so a failure there is rank-local and safe to swallow.
    try:
        if latents is not None:
            pipe.load_models_to_device(["vae"])
            video = pipe.vae.decode(latents.to(pipe.torch_dtype), device=pipe.device, tiled=False)
            video = pipe.vae_output_to_video(video)
            save_path = os.path.join(vis_dir, f"step_{step:06d}_rank{rank}.mp4")
            save_video(video, save_path, fps=16, quality=5)
            print(f"[vis] rank{rank} step {step} saved to {save_path}")
    except Exception as e:
        print(f"[vis] rank{rank} step {step} decode/save failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        # Has to run on every rank or their schedulers diverge for the next call.
        pipe.infer_scheduler.set_timesteps(1000, training=True)


# ─────────────────────────────────────────────────────────────
# Gradient checkpointing patch for DeepSpeed ZeRO-3
# ─────────────────────────────────────────────────────────────

def _patch_gradient_checkpointing():
    import deepspeed.runtime.activation_checkpointing.checkpointing as ds_ckpt

    _ds_checkpoint_fn = ds_ckpt.checkpoint

    def _compat_checkpoint(fn, *args, use_reentrant=None, **kwargs):
        if kwargs:
            fn = partial(fn, **kwargs)
        has_grad = any(isinstance(a, torch.Tensor) and a.requires_grad for a in args)
        if not has_grad:
            args = list(args)
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor) and a.is_floating_point():
                    args[i] = a.detach().requires_grad_(True)
                    break
            args = tuple(args)
        return _ds_checkpoint_fn(fn, *args)

    torch.utils.checkpoint.checkpoint = _compat_checkpoint


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def _optimizer_state_path(output_path):
    return os.path.join(output_path, "optimizer.pt")


def _save_optimizer_state(fsdp_model, optimizer, output_path, step, accelerator):
    """Write the AdamW moments so a restart does not re-warm them from zero.

    Only the latest is kept: resume always continues from the newest weight
    checkpoint, and the moments are ~4.9 GB (2 fp32 states over 613.7 M params).
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (FullOptimStateDictConfig,
                                        FullStateDictConfig, StateDictType)

    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        osd = FSDP.optim_state_dict(fsdp_model, optimizer)
    if accelerator.is_main_process:
        tmp = _optimizer_state_path(output_path) + ".tmp"
        torch.save({"step": step, "optimizer": osd}, tmp)
        os.replace(tmp, _optimizer_state_path(output_path))
        print(f"[optim] saved optimizer state at step {step}")


def _load_optimizer_state(fsdp_model, optimizer, output_path, expected_step, accelerator):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (FullOptimStateDictConfig,
                                        FullStateDictConfig, StateDictType)

    path = _optimizer_state_path(output_path)
    if not os.path.exists(path):
        if accelerator.is_main_process:
            print(f"[optim] no optimizer state at {path}, moments start from zero")
        return

    payload = None
    if accelerator.is_main_process:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("step") != expected_step:
            # The moments belong to a different point in training than the
            # weights being resumed; using them would be worse than not.
            print(f"[optim] {path} is at step {payload.get('step')} but resuming "
                  f"weights at step {expected_step}; ignoring it")
            payload = None

    use = torch.tensor([1 if payload is not None else 0], device=accelerator.device)
    torch.distributed.broadcast(use, src=0)
    if use.item() == 0:
        return

    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        osd = payload["optimizer"] if payload is not None else None
        osd = FSDP.optim_state_dict_to_load(fsdp_model, optimizer, osd)
        optimizer.load_state_dict(osd)
    if accelerator.is_main_process:
        print(f"[optim] restored optimizer state from step {expected_step}")


def launch_training(
    accelerator,
    dataloader,
    model,
    model_logger,
    args,
):
    lr = args.learning_rate
    weight_decay = getattr(args, "weight_decay", 1e-2)
    use_deepspeed = getattr(args, "use_deepspeed", False)
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    vis_every = getattr(args, "vis_every", 100)
    vis_dir = os.path.join(args.output_path, "vis")

    pipe = model.pipe

    if use_deepspeed:
        # ── DeepSpeed ZeRO-3：只对 dit 做 partition ──
        import deepspeed
        import deepspeed.runtime.activation_checkpointing.checkpointing as ds_ckpt
        import torch.distributed as dist

        # 验证多节点分布式环境（torchrun 已正确初始化 torch.distributed）
        assert dist.is_initialized(), (
            "torch.distributed is NOT initialized! Please use torchrun to launch."
        )
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"[dist check] rank={rank}, world_size={world_size}, "
              f"local_rank={accelerator.local_process_index}, backend={dist.get_backend()}")

        # frozen 模块先搬 GPU（不参与 DeepSpeed）
        pipe.device = accelerator.device
        for name in ("image_encoder", "vae", "text_encoder"):
            child = getattr(pipe, name, None)
            if child is not None and isinstance(child, torch.nn.Module):
                child.to(accelerator.device)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "betas": [0.9, 0.999],
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": lr,
                    "warmup_num_steps": 0,
                }
            },
            "bf16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 1e9,
                "stage3_prefetch_bucket_size": 1e9,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_max_live_parameters": 2e9,
                "stage3_max_reuse_distance": 2e9,
                "round_robin_gradients": True,
            },
            "gradient_clipping": 1.0,
            "communication_data_type": "bf16",
        }

        # deepspeed.initialize 复用 torchrun 已建立的 process group
        ds_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=pipe.dit,
            model_parameters=[p for p in pipe.dit.parameters() if p.requires_grad],
            config=ds_config,
        )
        pipe.dit = ds_engine
        _ds_engine = ds_engine

        # 配置 DeepSpeed activation checkpointing 并 monkey-patch
        ds_config["activation_checkpointing"] = {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": False,
        }
        ds_ckpt.configure(mpu_=None, deepspeed_config=ds_config)
        _patch_gradient_checkpointing()

        if accelerator.is_main_process:
            print(f"\n[train] DeepSpeed ZeRO-3: only pipe.dit is partitioned.")
            print(f"[train] Gradient checkpointing: DeepSpeed checkpoint + requires_grad fix.")
            print(f"[train] world_size={world_size}")
            total = sum(p.numel() for p in ds_engine.parameters() if p.requires_grad)
            print(f"[train] Trainable parameters in dit: {total / 1e6:.1f}M")

    else:
        # ── FSDP：只对 dit 做 wrap ──
        import torch.distributed as dist
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

        sharding_strategy_name = getattr(args, "fsdp_sharding_strategy", "hybrid")
        if sharding_strategy_name == "hybrid":
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        elif sharding_strategy_name == "full":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD

        # FSDP keeps the sharded master copy in the params' ORIGINAL dtype and
        # uses MixedPrecision.param_dtype only for the unsharded forward/backward
        # copy (torch/distributed/fsdp/_flat_param.py:579, 908). Leaving the DiT
        # in bf16 therefore gave bf16 master weights AND bf16 AdamW moments; with
        # 1-beta2=1e-3, exp_avg_sq increments round to zero once they fall ~3
        # decades below the running value, v collapses, and lr*m/sqrt(v) blows up
        # individual coordinates -- the run's LoRA absmax went 0.036 -> 54 while
        # its L2 only grew 74x, which is that signature and not a large gradient.
        # The upcast has to cover the whole module: a FlatParameter must have a
        # uniform dtype and use_orig_params only exempts requires_grad
        # (_flat_param.py:773-795), so the trainable tensors cannot be upcast
        # alone. Costs ~3.5 GB/GPU more for the sharded params.
        pipe.dit.to(torch.float32)

        wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(1e7))
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            # NOT keep_low_precision_grads=True. It would halve the sharded
            # gradient to 4.1 GB, but it leaves bf16 grads against fp32 params
            # and AdamW's foreach path requires a uniform dtype per group:
            # "Tensors of the same index must be on the same device and the same
            # dtype", raised on the first optimizer.step(). Measured separately:
            # forward+backward peaks at 66.9 GB with bf16 grads at 480x720 /
            # target_len 73, so fp32 grads still fit in the 79.7 GB card.
        )
        pipe.dit = FSDP(
            pipe.dit,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            device_id=accelerator.device,
            use_orig_params=True,
        )

        pipe.device = accelerator.device
        for name in ("image_encoder", "vae", "text_encoder"):
            child = getattr(pipe, name, None)
            if child is not None and isinstance(child, torch.nn.Module):
                child.to(accelerator.device)

        trainable_params = [p for p in pipe.dit.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        # NOT ConstantLR. Despite the name it is a 5-step warmup at base_lr/3
        # (factor=1/3, total_iters=5 by default) that multiplies the lr by 3 at
        # step 5. Combined with the load_state_dict below -- which restores
        # param_groups, lr included -- every restart inherited the previous
        # task's already-tripled lr and tripled it again. Three runs died this
        # way: each diverged in its third task, the first one to reach 9x.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        _ds_engine = None

        if getattr(args, "initial_step", 0) > 0:
            _load_optimizer_state(pipe.dit, optimizer, args.output_path,
                                  args.initial_step, accelerator)
            # The moments are worth restoring; the lr that came with them is not.
            for group in optimizer.param_groups:
                group["lr"] = lr
            if accelerator.is_main_process:
                print(f"[optim] lr forced back to the configured {lr:g} after restore")

        if accelerator.is_main_process:
            print(f"\n[train] Using FSDP ({sharding_strategy_name} shard), only pipe.dit is wrapped.")
            print(f"[train] world_size={dist.get_world_size()}")

    # ── Training loop ──
    global_step = getattr(args, "initial_step", 0)
    last_data = None
    # TokenBucketSampler iterates forever, so num_epochs never advances past 0
    # and max_steps is what actually ends the run.
    max_steps = getattr(args, "max_steps", None)

    for epoch_id in range(num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process,
                            desc=f"Epoch {epoch_id}", total=max_steps)

        for data in progress_bar:
            if max_steps is not None and global_step >= max_steps:
                break
            last_data = data

            if _ds_engine is not None:
                _ds_engine.zero_grad()
            else:
                optimizer.zero_grad()

            loss = model(data)

            grad_norm = None
            if _ds_engine is not None:
                _ds_engine.backward(loss)
                _ds_engine.step()
            else:
                loss.backward()
                # clip_grad_norm_ returns the total norm *before* clipping. The
                # first two runs diverged with no record of it, so there was no
                # way to tell a gradient spike from optimizer-state drift.
                grad_norm = pipe.dit.clip_grad_norm_(1.0)
                optimizer.step()
                scheduler.step()

            if accelerator.is_main_process:
                postfix = {"loss": f"{loss.item():.6f}", "step": global_step}
                if grad_norm is not None:
                    postfix["gnorm"] = f"{grad_norm.item():.3f}"
                # The lr that is actually in effect. It silently tripled on every
                # restart for three runs before anyone could see it, because the
                # only record of it was inside optimizer.pt.
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                progress_bar.set_postfix(**postfix)

            if vis_every > 0 and global_step % vis_every == 0 and last_data is not None:
                run_vis(
                    accelerator, model, last_data, vis_dir, global_step,
                    clean_x=args.vis_use_clean_x if args.vis_use_clean_x else None,
                )

            model_logger.on_step_end(accelerator, model, save_steps)
            if save_steps is not None and model_logger.num_steps % save_steps == 0 and _ds_engine is None:
                _save_optimizer_state(pipe.dit, optimizer, args.output_path,
                                      model_logger.num_steps, accelerator)
            global_step += 1

        if max_steps is not None and global_step >= max_steps:
            break

    model_logger.on_training_end(accelerator, model, save_steps)


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dataset_path", nargs="+", required=True)
    parser.add_argument("--dataset_weights", nargs="+", type=int, default=[1])
    parser.add_argument("--max_num_frames", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="./output_deepspeed")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--initial_step", type=int, default=0,
                        help="Step counter to resume from, so checkpoint numbering continues across restarts.")
    parser.add_argument("--dataset_type", type=str, default="animate", choices=["animate", "render"])
    parser.add_argument("--target_hw", type=str, default=None,
                        help="Resolution buckets as JSON, e.g. '[[480, 848]]'. Defaults to the built-in buckets.")
    parser.add_argument("--ref_frame_mode", type=str, default="window_first", choices=["window_first", "random"],
                        help="Which real frame feeds the in-context ref slot. Only used by --dataset_type render.")
    parser.add_argument("--dataset_scale", type=int, default=1,
                        help="Index-space multiplier for RandomMappingDataset. scale=1 leaves ~37%% of clips unreachable.")
    parser.add_argument("--target_fps", type=float, default=-1,
                        help="Resample every clip to this fps so a target_len window is the same duration everywhere. -1 keeps native fps.")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--vis_every", type=int, default=100)
    parser.add_argument("--vis_use_clean_x", action="store_true", default=False)
    parser.add_argument("--task", type=str, default="sft:causal_incontext_tsf")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--tf_clean_aug_max_timestep", type=int, default=0)
    parser.add_argument("--tf_sync_noise_to_clean", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dataset_num_workers", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--initialize_model_on_cpu", action="store_true", default=False)
    parser.add_argument("--trainable_models", type=str, default=None)
    parser.add_argument("--lora_base_model", type=str, default=None)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--extra_trainable_modules", type=str, default=None,
                        help="Comma-separated dit submodules to train alongside LoRA, e.g. 'pose_patch_embedding'.")
    parser.add_argument("--preset_lora_path", type=str, default=None)
    parser.add_argument("--preset_lora_model", type=str, default=None)
    parser.add_argument("--target_len", type=int, default=49)
    parser.add_argument("--local_attn_size", type=int, default=6)
    parser.add_argument("--tsf_ref_noise_timestep_id", type=int, default=None)
    parser.add_argument("--ref_noise_start_block", type=int, default=2)
    parser.add_argument("--bid_mode", action="store_true", default=False)
    parser.add_argument("--use_deepspeed", action="store_true", default=False,
                        help="Use DeepSpeed ZeRO instead of FSDP for dit sharding")
    parser.add_argument("--zero_stage", type=int, default=3, choices=[1, 2, 3],
                        help="ZeRO stage (only with --use_deepspeed)")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="hybrid",
                        choices=["hybrid", "full"],
                        help="FSDP sharding: hybrid=shard within node (fast), full=shard all GPUs")
    return parser


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    accel_kwargs = {}
    if not getattr(args, "use_deepspeed", False):
        accel_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    accelerator = accelerate.Accelerator(**accel_kwargs)

    model_paths = json.loads(args.model_paths)

    init_device = "cpu" if args.initialize_model_on_cpu else accelerator.device
    model = WanFullParamTrainingModule(
        model_paths=model_paths,
        tokenizer_path=args.tokenizer_path,
        device=init_device,
        task=args.task,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tf_clean_aug_max_timestep=args.tf_clean_aug_max_timestep,
        tf_sync_noise_to_clean=args.tf_sync_noise_to_clean,
        tsf_ref_noise_timestep_id=args.tsf_ref_noise_timestep_id,
        ref_noise_start_block=args.ref_noise_start_block,
        bid_mode=args.bid_mode,
        use_gradient_checkpointing=True,
        extra_trainable_modules=args.extra_trainable_modules,
    )
    model.pipe.dit.local_attn_size = args.local_attn_size
    if accelerator.is_main_process:
        print(f"local_attn_size={args.local_attn_size}, target_len={args.target_len}")

    if accelerator.is_main_process:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total / 1e9:.2f}B")

    dataset_kwargs = {
        "max_num_frames": args.max_num_frames,
        "dataset_max_token": 160000,
        "max_token": 160000,
        "max_wh": 720,
        "enable_motion_frames": True,
        "target_hw": [(512, 512), (720, 400), (400, 720)],
        "target_len": args.target_len,
        "motion_len": 2,
    }
    if args.target_hw is not None:
        dataset_kwargs["target_hw"] = [tuple(hw) for hw in json.loads(args.target_hw)]
    if args.dataset_type == "render":
        dataset_cls = RenderVideoDataset
        dataset_kwargs["ref_frame_mode"] = args.ref_frame_mode
        dataset_kwargs["dataset_scale"] = args.dataset_scale
        dataset_kwargs["target_fps"] = args.target_fps
    else:
        dataset_cls = AnimateVideoDataset
    dataset = dataset_cls.make_datasets(
        args.dataset_path, args, infer=False, **dataset_kwargs
    )
    sampler = TokenBucketSampler(
        dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_sampler=sampler,
        batch_size=1,
        num_workers=args.dataset_num_workers,
        pin_memory=True,
    )

    model_logger = ModelLogger(
        output_path=args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    model_logger.num_steps = args.initial_step

    launch_training(accelerator, dataloader, model, model_logger, args)

    # TokenBucketSampler spawns a daemon preload thread whose _stop_flag upstream
    # never sets. Before --max_steps the training loop never terminated, so the
    # process never reached a clean exit; now it does, and tearing down the
    # interpreter with that thread still live aborts with "terminate called
    # without an active exception" (SIGABRT) *after* all checkpoints are written,
    # which would make every successful run report FAILED.
    sampler._stop_flag = True
    sampler.thread.join(timeout=10)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if accelerator.is_main_process:
        print(f"[train] finished at step {model_logger.num_steps}")
