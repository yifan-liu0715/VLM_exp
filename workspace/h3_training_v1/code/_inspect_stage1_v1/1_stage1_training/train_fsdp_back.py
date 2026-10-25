"""
train_fsdp.py — LiveAnimate 全参数微调，支持 FSDP 单机多卡 / 多机多卡

用法（单机 8 卡）：
  bash train_fsdp.sh

用法（多机，每台机器分别执行）：
  MASTER_IP=<node0_ip> MACHINE_RANK=0 bash train_fsdp_multinode.sh
  MASTER_IP=<node0_ip> MACHINE_RANK=1 bash train_fsdp_multinode.sh
"""

import sys, os, json, math
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
from diffsynth.pipelines.wan_video_causal_animate_training_motion import WanVideoPipeline
from diffsynth.diffusion import *
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.utils.data import save_video, VideoData

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────────────────────
# Training module (全参数 / LoRA)
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
        bid_mode=False
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

        # 全参数微调：只冻结 VAE / text_encoder / image_encoder 以及 dit 内的 adapter 模块，DIT 主体可训
        # LoRA 微调：复用 DiffusionTrainingModule 的标准 LoRA 注入逻辑。
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
            "animate_face_video": data["tgt_frame_face"],
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
            # Optional knobs for task `sft:causal_incontext_tsf_refnoise`
            "tsf_ref_noise_timestep_id": self.tsf_ref_noise_timestep_id,
            "ref_noise_start_block": self.ref_noise_start_block,
            # Optional knobs for task `sft:bid_incontext_tsf`
            "bid_mode": self.bid_mode
        }
        return inputs_shared, inputs_posi, inputs_nega

    def _encode_prompt(self, prompt):
        """在 FSDP forward 上下文内调用 text_encoder，确保 all-gather 正常触发。"""
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
        else:
            return self.forward_train(data)
    
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
        
    def forward_train(self, data):
        inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        inputs_shared, inputs_posi, inputs_nega = inputs
        # 在 FSDP forward 上下文内做 text encoding，绕过 pipeline unit 的直接调用
        if "prompt" in inputs_posi and inputs_posi["prompt"] is not None:
            inputs_posi["context"] = self._encode_prompt(inputs_posi["prompt"])
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


# ─────────────────────────────────────────────────────────────
# Visualisation (every rank, every vis_every steps)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_vis(accelerator, model, data, vis_dir, step, clean_x=None):
    """Quick inference on the first sample of the batch, save to vis_dir/step_XXXXXX_rankN.mp4."""
    rank = accelerator.process_index
    os.makedirs(vis_dir, exist_ok=True)

    try:
        unwrapped = accelerator.unwrap_model(model)
        pipe = unwrapped.pipe

        sample = {k: (v[:1] if isinstance(v, torch.Tensor) else ([v[0]] if isinstance(v, list) else v))
                  for k, v in data.items()}

        # text_encoder 和 dit 都被 FSDP wrap，全部在 model.forward 上下文内调用
        latents = model(sample, vis=True, clean_x=clean_x)
        if latents is None:
            return

        # decode 和保存
        pipe.load_models_to_device(["vae"])
        video = pipe.vae.decode(latents.to(pipe.torch_dtype), device=pipe.device, tiled=False)
        video = pipe.vae_output_to_video(video)
        if rank in [0,1,2,3,4,5,6,7]:
            save_path = os.path.join(vis_dir, f"step_{step:06d}_rank{rank}.mp4")
            save_video(video, save_path, fps=16, quality=5)
            print(f"[vis] rank{rank} step {step} saved to {save_path}")

        pipe.infer_scheduler.set_timesteps(1000, training=True)
    except Exception as e:
        print(f"[vis] step {step} failed: {e}")
        import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# FSDP-aware training loop
# ─────────────────────────────────────────────────────────────

def launch_training_fsdp(
    accelerator,
    dataloader,
    model,
    model_logger,
    args,
):
    lr = args.learning_rate
    weight_decay = getattr(args, "weight_decay", 1e-2)
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    vis_every = getattr(args, "vis_every", 100)
    vis_dir = os.path.join(args.output_path, "vis")

    # 只对 dit 做 FSDP shard。
    # text_encoder / vae / image_encoder 均为 frozen，排除在 FSDP 之外，
    # prepare 后手动移到 GPU。text_encoder 在 pipeline unit 里被直接调用（不在 FSDP
    # forward 上下文内），若被 FSDP wrap 会导致 all-gather 时机错误，引发 device 不一致。
    if accelerator.distributed_type == accelerate.utils.DistributedType.FSDP:
        wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(1e7))
        accelerator.state.fsdp_plugin.auto_wrap_policy = wrap_policy

        ignored = []
        pipe = model.pipe
        for name in ("vae", "image_encoder", "tokenizer", "scheduler"):
            child = getattr(pipe, name, None)
            if child is not None and isinstance(child, torch.nn.Module):
                ignored.append(child)
        accelerator.state.fsdp_plugin.ignored_modules = ignored

    # Only pass trainable parameters to optimizer to avoid wasted optimizer state on frozen weights
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # prepare 之后更新 pipe.device，确保 pipeline 内所有 .to(pipe.device) 都指向正确的 GPU
    pipe = accelerator.unwrap_model(model).pipe
    pipe.device = accelerator.device

    # vae / image_encoder  ignored_modules，FSDP 不会自动移到 GPU，需手动搬运
    if accelerator.distributed_type == accelerate.utils.DistributedType.FSDP:
        for name in ("image_encoder", "vae"):
            child = getattr(pipe, name, None)
            if child is not None and isinstance(child, torch.nn.Module):
                child.to(accelerator.device)

    global_step = 0
    last_data = None

    for epoch_id in range(num_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process,
                            desc=f"Epoch {epoch_id}")
        for data in progress_bar:
            last_data = data
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=f"{loss.item():.6f}", step=global_step)

            # Visualisation
            if global_step % vis_every == 0 and last_data is not None:
                run_vis(
                    accelerator,
                    model,
                    last_data,
                    vis_dir,
                    global_step,
                    clean_x=args.vis_use_clean_x if args.vis_use_clean_x else None,
                )

            # Checkpoint
            model_logger.on_step_end(accelerator, model, save_steps)
            
            global_step += 1

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
    parser.add_argument("--output_path", type=str, default="./output_fsdp")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--vis_every", type=int, default=100)
    parser.add_argument("--vis_use_clean_x", action="store_true", default=False)
    parser.add_argument("--task", type=str, default="sft:causal_incontext_motion")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--tf_clean_aug_max_timestep", type=int, default=0)
    parser.add_argument("--tf_sync_noise_to_clean", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dataset_num_workers", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--find_unused_parameters", action="store_true", default=False)
    parser.add_argument("--initialize_model_on_cpu", action="store_true", default=False)
    parser.add_argument("--trainable_models", type=str, default=None)
    parser.add_argument("--lora_base_model", type=str, default=None)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--preset_lora_path", type=str, default=None)
    parser.add_argument("--preset_lora_model", type=str, default=None)
    parser.add_argument("--target_len", type=int, default=49)
    parser.add_argument("--local_attn_size", type=int, default=6)
    parser.add_argument(
        "--tsf_ref_noise_timestep_id",
        type=int,
        default=None,
        help="Fixed timestep id (t0) for ref noise in task sft:causal_incontext_tsf_refnoise; if omitted, sample a random id within [min_timestep_boundary, max_timestep_boundary) each step.",
    )
    parser.add_argument(
        "--ref_noise_start_block",
        type=int,
        default=2,
        help="Start using noisy ref from this temporal block (1-indexed). Default: 2 means block1 clean, block2+ noisy.",
    )
    parser.add_argument("--bid_mode", action="store_true", default=False)
    return parser


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

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
    dataset = AnimateVideoDataset.make_datasets(
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
    )

    default_remove_prefix = f"pipe.{args.lora_base_model}." if args.lora_base_model else "pipe.dit."
    remove_prefix_in_ckpt = args.remove_prefix_in_ckpt if args.remove_prefix_in_ckpt is not None else default_remove_prefix
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=remove_prefix_in_ckpt,
    )

    launch_training_fsdp(accelerator, dataloader, model, model_logger, args)
