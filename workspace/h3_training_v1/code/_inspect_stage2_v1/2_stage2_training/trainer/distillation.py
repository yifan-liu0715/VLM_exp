# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import gc
import logging
import random
import re
import torch
import wandb
import time
import os
import copy
from pathlib import Path
from omegaconf import OmegaConf
from einops import rearrange

import peft
from peft import get_peft_model_state_dict
import safetensors.torch

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)
from torchvision.io import write_video

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

############
from trainer.dmd import DMD
from trainer.lora_helper import add_lora_to_model, fuse_lora_to_base_model
from diffsynth.core import load_state_dict
from utils.checkpoint_utils import save_fsdp_checkpoint_with_topk
from utils.checkpoint_utils import (find_latest_resume_step, save_optimizer_state,
                                   load_lora_state, load_optimizer_state)
from utils.memory import log_gpu_memory
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (set_seed, merge_dict_list)
from datasets.data_v2v_pose import TokenBucketSampler, prepare_multimodal_batch, cycle
from datasets.data_v2v_pose import AnimateVideoDatasetWithPose as AnimateVideoDataset
from datasets.data_render import RenderVideoDataset
############


class Trainer:
    
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == -1:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb: # false
            wandb.login(
                key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")
        
        # ================================= LoRA Configuration =================================
        self.is_lora_enabled = False
        self.lora_config = None

        if hasattr(config, 'adapter') and config.adapter is not None:
            self.is_lora_enabled = True
            self.lora_config = config.adapter
            if self.is_main_process:
                print(f"LoRA enabled with config: {self.lora_config}")
                print("Loading base model and applying LoRA before FSDP wrapping...")
                
            # Freeze all model parameters
            self.model.generator.model.requires_grad_(False)
            self.model.fake_score.model.requires_grad_(False)
            self.model.text_encoder.requires_grad_(False)
            self.model.image_encoder.requires_grad_(False)
            self.model.vae.model.requires_grad_(False)

            pretrained_lora_path_bid = getattr(config, "bid_lora_ckpt", None)
            self.model.real_score = add_lora_to_model(
                self.model.real_score,
                lora_rank=self.lora_config.get('rank', 16),
                lora_target_modules=self.lora_config.get('lora_target_modules', "q,k,v,o,ffn.0,ffn.2"),
                pretrained_lora_path=pretrained_lora_path_bid,
            )
            self.model.real_score.model.requires_grad_(False)
            
            # Add LoRA layers to the generator and fake_score model
            pretrained_lora_path = getattr(config, "lora_ckpt", None)
            if pretrained_lora_path is None:
                print("Warning: No LoRA checkpoint specified, starting LoRA training from scratch")
            self.model.generator = add_lora_to_model(
                self.model.generator,
                lora_rank=self.lora_config.get('rank', 16),
                lora_target_modules=self.lora_config.get('lora_target_modules', "q,k,v,o,ffn.0,ffn.2"),
                pretrained_lora_path=pretrained_lora_path,
            )
            # Upstream leaves the critic's adapter at init, i.e. the raw base
            # model. DMD's gradient is roughly (s_fake - s_real); with the critic
            # unadapted and the teacher adapted, step 0's difference is dominated
            # by "the critic has never seen this task" rather than by any real
            # gap between the student's distribution and the teacher's. Starting
            # it from the same stage1 weights as the teacher and the student is
            # what DMD2-style implementations do. Set fake_lora_ckpt: null to
            # reproduce upstream.
            fake_lora_path = getattr(config, "fake_lora_ckpt", "__unset__")
            if fake_lora_path == "__unset__":
                fake_lora_path = pretrained_lora_path
            self.model.fake_score = add_lora_to_model(
                self.model.fake_score,
                lora_rank=self.lora_config.get('rank', 16),
                lora_target_modules=self.lora_config.get('lora_target_modules', "q,k,v,o,ffn.0,ffn.2"),
                pretrained_lora_path=fake_lora_path,
            )
            if self.is_main_process:
                print(f"[stage2] critic init from: {fake_lora_path}")

        # Resume BEFORE FSDP wraps anything. FSDP builds its root state lazily on
        # the first forward; calling state_dict APIs before that -- and on
        # `.model`, which is not the FSDP root -- sets `_is_root` on a non-root
        # instance and the next forward dies with "Non-root FSDP instance's
        # `_is_root` should not have been set yet". Here the modules are still
        # plain nn.Modules and the checkpoint's keys already match them.
        # checkpoint_save_dir is only assigned much later in __init__, well after
        # FSDP wrapping; establish it here so the pre-FSDP resume can find it.
        self.checkpoint_save_dir = os.path.join(self.output_path, "ckpt")
        self._resume_step = self._resume_weights_before_fsdp()

        # ============================================ FSDP wrapping ============================================
        self.model.generator = fsdp_wrap(
            self.model.generator, # causal wan
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        self.model.real_score = fsdp_wrap( 
            self.model.real_score, # wan
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score, # wan
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder, # text encoder
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
        
        # self.model.image_encoder = fsdp_wrap(
        #     self.model.image_encoder, # image encoder
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.image_encoder_fsdp_wrap_strategy,
        # ) # fsdp 有问题，是不是不fsdp可能比较好？
        
        self.model.image_encoder = self.model.image_encoder.to(
            device="cuda", dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # some modules are not wrapped with FSDP, move them to GPU manually
        self.model.vae = self.model.vae.to(
            device="cuda", dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        trainable_params = [param for param in self.model.generator.parameters() if param.requires_grad]
        total_size = sum(p.numel() * p.element_size() for p in trainable_params) / (1024 ** 2)
        print(f"generator可训练参数总大小: {total_size:.2f} MB")
        
        trainable_params = [param for param in self.model.fake_score.parameters() if param.requires_grad]
        total_size = sum(p.numel() * p.element_size() for p in trainable_params) / (1024 ** 2)
        print(f"critic可训练参数总大小: {total_size:.2f} MB")


        # Initialize optimizers with start LRs (fall back to end LRs if not set)
        generator_lr_start = getattr(config, "generator_lr_start", config.lr)
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=generator_lr_start,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        critic_lr_end = getattr(config, "lr_critic", config.lr)
        critic_lr_start = getattr(config, "critic_lr_start", critic_lr_end)
        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=critic_lr_start,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        self._resume_optimizers()

        # Step 5: Initialize the dataloader
        if self.config.v2v: # true
            v2v_training_config = config.v2v_training_config
            # `dataset_type: render` swaps the pose/face corpus for the low-poly
            # render one. The render dataset emits no `tgt_frame_face`, which the
            # rest of the pipeline now tolerates (face branch inert, pose branch
            # unaffected). `ref_frame_mode`/`dataset_scale`/`target_fps` are
            # render-only knobs and must not reach the pose dataset.
            ds_kwargs = dict(v2v_training_config.dataset)
            if getattr(v2v_training_config, "dataset_type", "animate") == "render":
                dataset_cls = RenderVideoDataset
                # RenderVideoDataset extends the plain AnimateVideoDataset, not
                # the WithPose subclass, so the pose/face-only knobs are not in
                # its signature and would raise TypeError.
                for k in ("enable_pose_keypoints", "pose_checkpoint",
                          "detector_checkpoint"):
                    ds_kwargs.pop(k, None)
            else:
                dataset_cls = AnimateVideoDataset
                for k in ("ref_frame_mode", "dataset_scale", "target_fps"):
                    ds_kwargs.pop(k, None)
            dataset = dataset_cls.make_datasets(
                v2v_training_config.dataset_path, v2v_training_config,
                infer=v2v_training_config.infer, **ds_kwargs)
            batch_sampler = TokenBucketSampler(dataset, rank=global_rank, world_size=self.world_size)
            self.batch_sampler = batch_sampler   # retained so train_distillation.py can stop its thread
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=v2v_training_config.dataloader_num_workers)
        else:
            raise ValueError("Invalid training task")

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader) # infinite dataloader，使用 cycle 后，可以在训练循环中直接使用 next(self.dataloader)

        # Step 6: visualization for validation (fixed prompts)
        # TODO : implement later if needed
        ##############################################################################################################
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None

        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")
            if LOG_GPU_MEMORY:
                log_gpu_memory("After initialization", device=self.device, rank=dist.get_rank())

        # Initialize checkpoint management for new save system
        self.saved_checkpoints = []  # 用于 top_k 管理
        self.checkpoint_save_dir = os.path.join(self.output_path, "ckpt")  # same value as above
        if self.is_main_process:
            os.makedirs(self.checkpoint_save_dir, exist_ok=True)
        
    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # ----------------------------- LR Scheduler (constant -> linear -> constant) -----------------------------
    def _compute_linear_lr(self, step, start, end, start_step, end_step):
        # No scheduling configured: use end value to preserve current behavior
        if end_step is None or end_step <= 0:
            return end
        # Normalize start_step
        if start_step is None or start_step < 0:
            start_step = 0
        # Degenerate window: treat as step function at start_step
        if end_step <= start_step:
            return start if step < start_step else end
        # Before window: keep start value
        if step <= start_step:
            return start
        # After window: keep end value
        if step >= end_step:
            return end
        # Linear interpolation within window [start_step, end_step]
        r = float(step - start_step) / float(end_step - start_step)
        return start + (end - start) * r

    def _apply_lr_schedule(self, step):
        end_step = getattr(self.config, "lr_scheduler_end_step", 0)
        start_step = getattr(self.config, "lr_scheduler_start_step", 0)

        # generator
        gen_start = getattr(self.config, "generator_lr_start", getattr(self.config, "lr", 0.0))
        gen_end = getattr(self.config, "lr", gen_start)
        lr_g = self._compute_linear_lr(step, gen_start, gen_end, start_step, end_step)
        for g in self.generator_optimizer.param_groups:
            g["lr"] = lr_g

        # critic (fake_score)
        crit_end = getattr(self.config, "lr_critic", getattr(self.config, "lr", 0.0))
        crit_start = getattr(self.config, "critic_lr_start", crit_end)
        lr_c = self._compute_linear_lr(step, crit_start, crit_end, start_step, end_step)
        for g in self.critic_optimizer.param_groups:
            g["lr"] = lr_c

    def get_all_checkpoints(self, logdir):
        """Get all checkpoints in the logdir sorted by step number."""
        if not os.path.exists(logdir):
            return []
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_dir_path = os.path.join(logdir, item)
                    checkpoint_file_path = os.path.join(checkpoint_dir_path, "model.pt")
                    if os.path.exists(checkpoint_file_path):
                        checkpoint_dirs.append((step, checkpoint_dir_path, item))
                except ValueError:
                    continue
        
        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def cleanup_old_checkpoints(self, logdir, max_checkpoints):
        """Remove old checkpoints if the number exceeds max_checkpoints.
        
        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return
        
        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return
            
        checkpoints = self.get_all_checkpoints(logdir)
        if len(checkpoints) > max_checkpoints:
            # Calculate how many to remove
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones
            
            print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})")
            
            import shutil
            removed_count = 0
            for step, checkpoint_dir_path, dir_name in checkpoints_to_remove:
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    shutil.rmtree(checkpoint_dir_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")
            
            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")
        else:
            if len(checkpoints) > 0:
                print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)")


    def save(self):
        print("Start gathering distributed model states...")
        if self.is_lora_enabled:
            # 使用新的保存函数来保存 LoRA 检查点
            max_checkpoints = getattr(self.config, "max_checkpoints", 50)
            if self.is_main_process:
                print(f"Saving LoRA checkpoints at step {self.step}")
            # 保存 generator LoRA
            self.saved_checkpoints = save_fsdp_checkpoint_with_topk(
                model=self.model.generator.model,
                save_dir=self.checkpoint_save_dir,
                global_step=self.step,
                only_trainable=True,
                top_k=max_checkpoints,
                saved_checkpoints_list=self.saved_checkpoints,
                prefix="generator_lora"
            )
            
            # AdamW moments too. Without them every restart re-warms the
            # optimizer, and over a 5000-step run split into 2 h tasks that is
            # ~25 cold starts.
            # The FSDP root, matching _resume_optimizers(). Passing `.model` here
            # made the saved FQNs relative to the submodule ("blocks.0.self_attn
            # .q.lora_A...") while the loader, correctly working on the root,
            # looked for "model.blocks.0..." -> KeyError -> moments silently
            # re-warmed from zero on every restart.
            save_optimizer_state(self.model.generator, self.generator_optimizer,
                                 self.checkpoint_save_dir, self.step, "generator_lora")
            save_optimizer_state(self.model.fake_score, self.critic_optimizer,
                                 self.checkpoint_save_dir, self.step, "critic_lora")

            # 保存 critic LoRA
            self.saved_checkpoints = save_fsdp_checkpoint_with_topk(
                model=self.model.fake_score.model,
                save_dir=self.checkpoint_save_dir,
                global_step=self.step,
                only_trainable=True,
                top_k=max_checkpoints,
                saved_checkpoints_list=self.saved_checkpoints,
                prefix="critic_lora"
            )
            
            if self.is_main_process:
                print(f"LoRA checkpoints saved to {self.checkpoint_save_dir}")
        else:
            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),          # newly added
            ):
                generator_state_dict  = self.model.generator.state_dict()
                generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                                self.generator_optimizer)

            with FSDP.state_dict_type(
                self.model.fake_score,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),          # newly added
            ):
                critic_state_dict  = self.model.fake_score.state_dict()
                critic_opim_state_dict = FSDP.optim_state_dict(self.model.fake_score,
                                                self.critic_optimizer)

            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_optimizer": generator_opim_state_dict,
                "critic_optimizer": critic_opim_state_dict,
                "step": self.step,
            }

            if self.is_main_process:
                checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
                torch.save(state_dict, checkpoint_file)
                print("Model saved to", checkpoint_file)
                
                # Cleanup old checkpoints if max_checkpoints is set
                max_checkpoints = getattr(self.config, "max_checkpoints", 0) #100
                if max_checkpoints > 0:
                    self.cleanup_old_checkpoints(self.output_path, max_checkpoints)

        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    
    def fwdbwd_one_step(self, batch, train_generator, train_critic, accumulation_step=0):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # if self.step % 3 == 0:
        torch.cuda.empty_cache()

        # Step 1: Get the next batch of data batch
        if self.config.v2v: # true
           batch_size, cond_dict, neg_cond_dict, image_or_video_shape = prepare_multimodal_batch(batch,self.device,self.dtype,neg_cond_prompt=self.config.negative_prompt)
        else:
            raise ValueError("Invalid training task")
            
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # text
            context = self.model.text_encoder(text_prompts=cond_dict["prompt"], context=None)
            neg_context = self.model.text_encoder(text_prompts=neg_cond_dict["prompt"], context=None)
            # pose
            pose = self.model.vae.encode_to_latent(cond_dict["pose"]) # B f_lat 16 h_lat w_lat
            pose = pose[:,1:] # !!!!
            psi_pose = rearrange(pose,'b t c h w -> b c t h w')
            neg_pose = psi_pose
            # face -- absent on a face-less corpus; the pose/render branch above
            # is what this project conditions on and it stays active either way.
            if cond_dict.get("face", None) is not None:
                cond_dict["face"] = cond_dict["face"][:,:,1:]
            # keypoints: align with pose/face, drop the first frame
            if "keypoints" in cond_dict:
                cond_dict["keypoints"] = {
                    k: v[:, 1:] for k, v in cond_dict["keypoints"].items()
                }
            # clip_fea
            clip_rea = self.model.image_encoder(images=cond_dict["ref_image"])
            neg_clip_rea = clip_rea
            
            image_lat = self.model.vae.encode_to_latent(cond_dict["ref_image"].unsqueeze(2))  # B 1 C H W

            # y for causal_model
            padding_image = torch.zeros_like(cond_dict["ref_image"].unsqueeze(2))  # B c 1 h_lat w_lat
            padding_content = torch.zeros_like(cond_dict["pose"])  # B C f_lat h_lat w_lat
            padding_image_lat = self.model.vae.encode_to_latent(padding_image.to(dtype=self.dtype, device=self.device)) # B f_lat 16 h_lat w_lat
            padding_content_lat = self.model.vae.encode_to_latent(padding_content.to(dtype=self.dtype, device=self.device)) # B f_lat 16 h_lat w_lat
            padding_content_lat = padding_content_lat[:,1:]
            y = torch.concat([padding_image_lat, padding_content_lat], dim=1)
            y = rearrange(y,'b t c h w -> b c t h w')
            _,_,T,H,W = y.shape
            msk = torch.zeros(1, 4, T, H, W, device=self.device, dtype=self.dtype)
            y = y.to(dtype=self.dtype, device=self.device)
            y = torch.concat([msk, y], dim=1)
            y = y.to(dtype=self.dtype, device=self.device)

            # final cond dict
            cond_dict_input = {}
            cond_dict_input["face_pixel_values"] = (cond_dict["face"].to(self.device)
                                          if cond_dict.get("face", None) is not None else None)
            cond_dict_input["context"] = context.to(self.device)
            cond_dict_input["pose_latents"] = psi_pose.to(self.device)
            cond_dict_input["clip_feature"] = clip_rea.to(self.device)
            cond_dict_input["y"] = y.to(self.device)
            cond_dict_input["sink_flag"] = True
            cond_dict_input["ref_lat"] = image_lat.to(self.device)

            cond_dict_input["y_for_bid"] = y.to(self.device)
            cond_dict_input["pose_latents_for_bid"] = psi_pose.to(self.device)
            cond_dict_input["face_pixel_values_for_bid"] = (cond_dict["face"].to(self.device)
                                          if cond_dict.get("face", None) is not None else None)
            if "keypoints" in cond_dict:
                cond_dict_input["keypoints"] = {
                    k: v.to(self.device) for k, v in cond_dict["keypoints"].items()
                }
                cond_dict_input["keypoints_for_bid"] = {
                    k: v.to(self.device) for k, v in cond_dict["keypoints"].items()
                }

            # final neg cond dict
            neg_cond_dict_input = {}
            neg_cond_dict_input["face_pixel_values"] = (cond_dict["face"].to(self.device)
                                          if cond_dict.get("face", None) is not None else None)
            neg_cond_dict_input["context"] = neg_context.to(self.device)
            neg_cond_dict_input["pose_latents"] = neg_pose.to(self.device)
            neg_cond_dict_input["clip_feature"] = neg_clip_rea.to(self.device)
            neg_cond_dict_input["y"] = y.to(self.device)
            neg_cond_dict_input["sink_flag"] = True
            neg_cond_dict_input["ref_lat"] = image_lat.to(self.device) # B 1 16 h_lat w_lat

            neg_cond_dict_input["y_for_bid"] = y.to(self.device)
            neg_cond_dict_input["pose_latents_for_bid"] = neg_pose.to(self.device)
            neg_cond_dict_input["face_pixel_values_for_bid"] = (cond_dict["face"].to(self.device)
                                          if cond_dict.get("face", None) is not None else None)


        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            if LOG_GPU_MEMORY:
                log_gpu_memory("Before train_generator forward pass", device=self.device, rank=dist.get_rank())
            
            # if dist.get_rank() == 0:
            #     import pdb; pdb.set_trace()
            
            # Interval was hardcoded to 50. Each visualisation decodes and writes a
            # video on all 8 ranks, so at 50 it fires 400 times over a 20k-step run.
            if self.step % getattr(self.config, "vis_interval", 50) == 0 and accumulation_step == 0:
                VISUALIZE = True
            else:
                VISUALIZE = False
            
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape, # 1 21 16 90 50
                conditional_dict=cond_dict_input, # conditional info 
                unconditional_dict=neg_cond_dict_input, # unconditional 
                clean_latent=None,
                initial_latent=None,
                accumulation_step=accumulation_step, # the index if current accumulation step (block indx for gradient calculation)
                step=self.step,
                visualization=VISUALIZE
            )
            
            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps # 
            if LOG_GPU_MEMORY:
                log_gpu_memory(f"Before train_generator backward pass at accumulation_step {accumulation_step}", device=self.device, rank=dist.get_rank())
            # with (self.model.generator.no_sync() if accumulation_step!=3 else nullcontext()):
            torch.cuda.reset_peak_memory_stats(self.device)

            scaled_generator_loss.backward()
            
            peak = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            print(f"loss backward - current: {torch.cuda.memory_allocated(self.device) / 1024 / 1024:.2f} MB, peak: {peak:.2f} MB")
            self.model.inference_pipeline._free_cache_memory(accumulation_step, num_blocks_clip=self.gradient_accumulation_steps)
            
            if LOG_GPU_MEMORY:
                log_gpu_memory(f"After train_generator backward pass at accumulation_step {accumulation_step}", device=self.device, rank=dist.get_rank())
            # Return original loss for logging
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

            return generator_log_dict
        else:
            generator_log_dict = {}
        
        if train_critic:
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=cond_dict_input,
                unconditional_dict=neg_cond_dict_input,
                clean_latent=None,
                initial_latent=None,
                step=self.step,
                visualization=False
            )

            # Scale loss for gradient accumulation and backward
            scaled_critic_loss = critic_loss
            scaled_critic_loss.backward()
            if LOG_GPU_MEMORY:
                log_gpu_memory("After train_critic backward pass", device=self.device, rank=dist.get_rank())
            # Return original loss for logging
            critic_log_dict.update({"critic_loss": critic_loss,
                                    "critic_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation
        else:
            critic_log_dict = {}

        return critic_log_dict
    

    def _resume_weights_before_fsdp(self):
        """Restore generator/critic LoRA into the unwrapped modules. Returns the step."""
        step = find_latest_resume_step(self.checkpoint_save_dir)
        if step is None:
            if self.is_main_process:
                print("[Resume] no complete checkpoint found, starting from step 0")
            return 0
        for mod, prefix in ((self.model.generator.model, "generator_lora"),
                            (self.model.fake_score.model, "critic_lora")):
            path = os.path.join(self.checkpoint_save_dir, f"step_{step}_{prefix}.pt")
            sd = torch.load(path, map_location="cpu", weights_only=False)
            missing, unexpected = mod.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"{path}: {len(unexpected)} unexpected keys, e.g. {list(unexpected)[:4]}")
            if len(sd) == 0:
                raise RuntimeError(f"{path} is empty; resume would silently do nothing.")
            if self.is_main_process:
                print(f"[Resume] {os.path.basename(path)}: {len(sd)} tensors restored (pre-FSDP)")
        if self.is_main_process:
            print(f"[Resume] continuing from step {step}")
        return step

    def _resume_optimizers(self):
        """Moments, after the optimizers exist. Best-effort: losing them costs a
        re-warm, whereas letting an FSDP state-dict quirk kill the job costs the
        whole run."""
        step = getattr(self, "_resume_step", 0)
        self.step = step
        if not step:
            return
        # The FSDP ROOT (self.model.generator), not `.model` underneath it. This
        # runs before the first forward, so FSDP has not lazily initialised yet;
        # entering state_dict_type on a NON-root submodule at this point sets
        # `_is_root` on a child and the first forward then dies with
        # "Non-root FSDP instance's `_is_root` should not have been set yet".
        # save_optimizer_state() is called on the root too, so the FQNs match.
        for model, opt, prefix, lr_key in (
                (self.model.generator, self.generator_optimizer, "generator_lora", "generator_lr_start"),
                (self.model.fake_score, self.critic_optimizer, "critic_lora", "critic_lr_start")):
            path = os.path.join(self.checkpoint_save_dir, f"step_{step}_{prefix}_optim.pt")
            try:
                load_optimizer_state(model, opt, path, step)
            except Exception as e:
                print(f"[Resume] optimizer restore for {prefix} failed ({e}); moments start from zero")
            # load_state_dict restores the WHOLE param_group, lr included -- that
            # is exactly how stage1's lr silently tripled on every restart.
            want = float(getattr(self.config, lr_key))
            for g in opt.param_groups:
                g["lr"] = want
        if self.is_main_process:
            print(f"[Resume] lrs pinned to {self.config.generator_lr_start:g}/{self.config.critic_lr_start:g}")

    def _unused_maybe_resume(self):
        """Pick up from the newest complete checkpoint pair in the ckpt dir.

        Upstream has no resume at all -- `self.step` is hardcoded to 0 -- so an
        array of 2 h tasks would restart from scratch every time and never get
        past the first ~200 steps. Weights and moments are restored; the frozen
        pose_patch_embedding is not in these files and already came from the
        stage1 checkpoint at construction.
        """
        step = find_latest_resume_step(self.checkpoint_save_dir)
        if step is None:
            if self.is_main_process:
                print("[Resume] no complete checkpoint found, starting from step 0")
            return
        d = self.checkpoint_save_dir
        load_lora_state(self.model.generator.model, os.path.join(d, f"step_{step}_generator_lora.pt"))
        load_lora_state(self.model.fake_score.model, os.path.join(d, f"step_{step}_critic_lora.pt"))
        load_optimizer_state(self.model.generator.model, self.generator_optimizer,
                             os.path.join(d, f"step_{step}_generator_lora_optim.pt"), step)
        load_optimizer_state(self.model.fake_score.model, self.critic_optimizer,
                             os.path.join(d, f"step_{step}_critic_lora_optim.pt"), step)
        # load_state_dict restores the WHOLE param_group, lr included. In stage1
        # that is exactly how the lr silently tripled on every restart. The lrs
        # here are constants from the config, so pin them back explicitly.
        for opt, key in ((self.generator_optimizer, "generator_lr_start"),
                         (self.critic_optimizer, "critic_lr_start")):
            want = float(getattr(self.config, key))
            for g in opt.param_groups:
                g["lr"] = want
        self.step = step
        if self.is_main_process:
            print(f"[Resume] continuing from step {step}, lrs pinned to "
                  f"{self.config.generator_lr_start:g}/{self.config.critic_lr_start:g}")

    def train(self):

        start_step = self.step
        try:
            while True:
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
                if LOG_GPU_MEMORY:
                    log_gpu_memory(f"Before training", device=self.device, rank=dist.get_rank())
                
                if TRAIN_GENERATOR:
                    self.generator_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                # Apply LR schedule for current step before any optimizer.step()
                # self._apply_lr_schedule(self.step)
                
                # Whole-cycle gradient accumulation loop
                accumulated_generator_logs = []
                accumulated_critic_logs = []
                
                batch = next(self.dataloader)
                if TRAIN_GENERATOR:# Train generator (accumulation steps) + critic (0 step)
                    TRAIN_CRITIC = False
                    for accumulation_step in range(self.gradient_accumulation_steps): # 7
                        extra_gen = self.fwdbwd_one_step(batch, True, TRAIN_CRITIC, accumulation_step) 
                        TRAIN_CRITIC = False
                        accumulated_generator_logs.append(extra_gen)
                else: # Train critic (1 step)
                    # Train critic
                    extra_crit = self.fwdbwd_one_step(batch, False, True) #只有critic update
                    accumulated_critic_logs.append(extra_crit)
                
                # Compute grad norm and update parameters
                if TRAIN_GENERATOR: # Train generator (accumulation steps) + critic (1 step)
                    generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                    generator_log_dict = merge_dict_list(accumulated_generator_logs)
                    generator_log_dict["generator_grad_norm"] = generator_grad_norm
                
                    self.generator_optimizer.step()

                else: # Train critic (1 step)
                    generator_log_dict = {}
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm

                    self.critic_optimizer.step()
                

                # Increment the step since we finished gradient update
                self.step += 1

                # Save the model
                if (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()

                # Logging
                if self.is_main_process:
                    wandb_loss_dict = {}
                    if TRAIN_GENERATOR and generator_log_dict:
                        wandb_loss_dict.update(
                            {
                                "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                                "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                                "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                            }
                        )

                    else:
                        wandb_loss_dict.update(
                            {
                                "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                                "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                            }
                        )
                    # Log current learning rates
                    try:
                        gen_lr = self.generator_optimizer.param_groups[0]["lr"]
                        crit_lr = self.critic_optimizer.param_groups[0]["lr"]
                        wandb_loss_dict.update({
                            "lr/generator": gen_lr,
                            "lr/critic": crit_lr,
                        })
                    except Exception:
                        pass
                    if not self.disable_wandb:
                        wandb.log(wandb_loss_dict, step=self.step)

                if self.step % self.config.gc_interval == 0:
                    if dist.get_rank() == 0:
                        logging.info("DistGarbageCollector: Running GC.")
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.is_main_process:
                    current_time = time.time()
                    iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    self.previous_time = current_time
                    # Log training progress
                    if TRAIN_GENERATOR and generator_log_dict:
                        # print(f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}, dmdtrain_gradient_norm {generator_log_dict['dmdtrain_gradient_norm'].mean().item()}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")
                        print(f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}, dmdtrain_gradient_norm {generator_log_dict['dmdtrain_gradient_norm'].mean().item()}")
                    else:
                        print(f"step {self.step}, per iteration time {iteration_time}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")

                
                if self.step > self.config.max_iters:
                    break

        except Exception as e:
            # Print on EVERY rank: gating on is_main_process meant a failure on
            # any other rank produced no output at all.
            print(f"[ERROR] rank {dist.get_rank()} crashed at step {self.step}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Re-raise. Swallowing made train() return normally, so the only
            # symptom left was the daemon-thread SIGABRT at interpreter teardown
            # -- which is how a plain "PyAV is not installed" ImportError looked
            # like an unexplained C++ abort for three probe rounds.
            raise
