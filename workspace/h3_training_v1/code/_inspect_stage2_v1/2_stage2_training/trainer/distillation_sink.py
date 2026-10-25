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
from trainer.dmd_sink import DMD
from trainer.lora_helper import add_lora_to_model, fuse_lora_to_base_model
from diffsynth.core import load_state_dict
from utils.checkpoint_utils import save_fsdp_checkpoint_with_topk
from utils.memory import log_gpu_memory
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (set_seed, merge_dict_list)
from datasets.data_v2v_diffface import AnimateVideoDataset, TokenBucketSampler, prepare_multimodal_batch, cycle
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
            self.model.fake_score = add_lora_to_model(
                self.model.fake_score,
                lora_rank=self.lora_config.get('rank', 16),
                lora_target_modules=self.lora_config.get('lora_target_modules', "q,k,v,o,ffn.0,ffn.2"),
                pretrained_lora_path=None,
            )

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

        # Step 5: Initialize the dataloader
        if self.config.v2v: # true
            v2v_training_config = config.v2v_training_config
            dataset = AnimateVideoDataset.make_datasets(v2v_training_config.dataset_path, v2v_training_config, infer=v2v_training_config.infer, **v2v_training_config.dataset)
            batch_sampler = TokenBucketSampler(dataset, rank=global_rank, world_size=self.world_size)
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
        self.checkpoint_save_dir = os.path.join(self.output_path, "ckpt")
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
            # face
            face_for_causal = cond_dict["face"][:,:,1:]
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

            # # y for bidirectional model (teacher)
            # # ref_img + zero content pixels -> VAE encode, 与 test_teacher_vis.py 对齐
            # image_lat = self.model.vae.encode_to_latent(cond_dict["ref_image"].unsqueeze(2))  # B 1 C H W
            # B_bid = image_lat.shape[0]
            # _, _, _, H_lat, W_lat = image_lat.shape
            # # content latent 帧数：从 pose 推算（pose 是 pixel space B C T H W）
            # T_content = (cond_dict["pose"].shape[2] - 1) // 4  # latent T (drop first VAE frame), 21
            # T_pixel_bid = T_content * 4 # 21 * 4 = 84
            # padding_pixels_bid = torch.zeros(
            #     B_bid, 3, T_pixel_bid, H_lat * 8, W_lat * 8,
            #     device=self.device, dtype=self.dtype
            # )
            # ref_img_bid = cond_dict["ref_image"].to(device=self.device, dtype=self.dtype)  # B C H W
            # ref_img_bid = ref_img_bid.unsqueeze(2)  # B C 1 H W
            # padding_pixels_bid = torch.cat([ref_img_bid, padding_pixels_bid], dim=2)  # B C 1+T_pixel H W # 85
            # y_vae_bid = self.model.vae.encode_to_latent(padding_pixels_bid)  # B T_lat C H W (encode_to_latent 接收 B C T H W) 1 + 21 = 22
            # y_vae_bid = rearrange(y_vae_bid, 'b t c h w -> b c t h w')
            # _, _, T_total_bid, Hy_bid, Wy_bid = y_vae_bid.shape
            # msk_bid = torch.ones(B_bid, 4, T_total_bid, Hy_bid, Wy_bid, device=self.device, dtype=self.dtype)
            # msk_bid[:, :, 1:] = 0
            # y_for_bid = torch.cat([msk_bid, y_vae_bid], dim=1).to(dtype=self.dtype, device=self.device)
            y_for_bid = y

            # # pose_latents for teacher: 完整 encode，teacher 用全部帧
            # pose_for_bid_pixel = cond_dict["pose"]  # B C T H W (pixel space, [-1,1])
            # pose_for_bid_pixel = pose_for_bid_pixel[:,:,4:]
            # pose_lat_bid = self.model.vae.encode_to_latent(pose_for_bid_pixel)  # B T_lat C H W
            # pose_latents_for_bid = rearrange(pose_lat_bid, 'b t c h w -> b c t h w')
            pose_latents_for_bid = psi_pose

            # # face for teacher: 完整，teacher 用全部帧
            # face_for_bid = cond_dict["face"][:,:,4:]  # B C T H W
            face_for_bid = face_for_causal

            # final cond dict
            cond_dict_input = {}
            cond_dict_input["face_pixel_values"] = face_for_causal.to(self.device)
            cond_dict_input["context"] = context.to(self.device)
            cond_dict_input["pose_latents"] = psi_pose.to(self.device)
            cond_dict_input["clip_feature"] = clip_rea.to(self.device)
            cond_dict_input["y"] = y.to(self.device)
            cond_dict_input["sink_flag"] = True
            cond_dict_input["ref_lat"] = image_lat.to(self.device)

            cond_dict_input["y_for_bid"] = y_for_bid.to(self.device)
            cond_dict_input["pose_latents_for_bid"] = pose_latents_for_bid.to(self.device)
            cond_dict_input["face_pixel_values_for_bid"] = face_for_bid.to(self.device)
            if "keypoints" in cond_dict:
                cond_dict_input["keypoints"] = {
                    k: v.to(self.device) for k, v in cond_dict["keypoints"].items()
                }
                cond_dict_input["keypoints_for_bid"] = {
                    k: v.to(self.device) for k, v in cond_dict["keypoints"].items()
                }

            # final neg cond dict
            neg_cond_dict_input = {}
            neg_cond_dict_input["face_pixel_values"] = (face_for_causal * 0 - 1.).to(self.device)
            neg_cond_dict_input["context"] = neg_context.to(self.device)
            neg_cond_dict_input["pose_latents"] = neg_pose.to(self.device)
            neg_cond_dict_input["clip_feature"] = neg_clip_rea.to(self.device)
            neg_cond_dict_input["y"] = y.to(self.device)
            neg_cond_dict_input["sink_flag"] = True
            neg_cond_dict_input["ref_lat"] = image_lat.to(self.device) # B 1 16 h_lat w_lat

            neg_cond_dict_input["y_for_bid"] = y_for_bid.to(self.device)
            neg_cond_dict_input["pose_latents_for_bid"] = pose_latents_for_bid.to(self.device)
            neg_cond_dict_input["face_pixel_values_for_bid"] = (face_for_bid * 0 - 1.).to(self.device)


        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            if LOG_GPU_MEMORY:
                log_gpu_memory("Before train_generator forward pass", device=self.device, rank=dist.get_rank())
            
            # if dist.get_rank() == 0:
            #     import pdb; pdb.set_trace()
            
            if self.step % 50 == 0 and accumulation_step == 0:
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
                        # if accumulation_step<2:
                        #     extra_gen = self.fwdbwd_one_step(batch, True, TRAIN_CRITIC, accumulation_step) 

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
            if self.is_main_process:
                print(f"[ERROR] Training crashed at step {self.step} with exception: {e}")
                print(f"[ERROR] Exception traceback:", flush=True)
                import traceback
                traceback.print_exc()
