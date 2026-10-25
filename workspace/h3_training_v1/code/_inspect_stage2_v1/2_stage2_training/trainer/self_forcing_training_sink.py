# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from functools import cache
from this import d
from turtle import backward, forward
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.training_option import BASE_MODEL, TRAINING_SCALE
from utils.memory import log_gpu_memory
from diffsynth.wan_wrapper_sink import WanDiffusionWrapper
from diffsynth.scheduler import SchedulerInterface
import copy
import random

class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 rerun_kv_cache: bool = False,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.vae = kwargs.get("vae", None)
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        if BASE_MODEL == "14B":
            self.num_transformer_blocks = 40
        else:
            self.num_transformer_blocks = 30

        # frame_seq_length: 用所有 bucket 中最大的分辨率来分配 KV cache，
        # 保证不同 bucket 的样本都能放进同一个 cache。
        # 支持通过 kwargs 传入 target_hw_buckets（list of (h,w)）来自动计算。
        target_hw_buckets = kwargs.get("target_hw_buckets", None)
        if target_hw_buckets is not None and len(target_hw_buckets) > 0:
            max_tokens = max((h // 16) * (w // 16) for h, w in target_hw_buckets)
            self.frame_seq_length = max_tokens
        elif TRAINING_SCALE == "720*400":
            self.frame_seq_length = 720 // 16 * 400 // 16  # 1125
        else:
            self.frame_seq_length = 1560
            
        self.num_frame_per_block = num_frame_per_block # 3
        self.rerun_kv_cache = rerun_kv_cache # trues
        self.context_noise = context_noise

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        self.independent_first_frame = independent_first_frame # false
        self.same_step_across_blocks = same_step_across_blocks # false
        self.last_step_only = last_step_only # false
        
        
        # Support local_attn_size as int or list (scheduled by timestep); compute KV cache frames internally
        self.local_attn_size = kwargs.get("local_attn_size", -1) # 12
        if not isinstance(self.local_attn_size, int) and hasattr(self.local_attn_size, "__iter__"):
            self.local_attn_size = list(self.local_attn_size)
            
        if isinstance(self.local_attn_size, (list, tuple)):
            assert len(self.local_attn_size) == len(self.denoising_step_list), (
                f"local_attn_size length ({len(self.local_attn_size)}) must match denoising_step_list length ({len(self.denoising_step_list)})."
            )
            if DEBUG:
                print(f"local_attn_size schedule length: {len(self.local_attn_size)}, denoising steps: {len(self.denoising_step_list)}")
        else:
            if DEBUG:
                print(f"Using static local_attn_size: {self.local_attn_size}")

        # Context used for KV cache calculation
        num_training_frames: Optional[int] = kwargs.get("num_training_frames", 21)
        slice_last_frames: int = int(kwargs.get("slice_last_frames", 21))

        # Compute KV cache supporting list/int and global attention (-1)
        def _resolve_kv_frames(local_cfg):
            if isinstance(local_cfg, (list, tuple)):
                base = int(max(local_cfg)) if len(local_cfg) > 0 else -1
                return min(base + slice_last_frames, num_training_frames)
            else:
                base = int(local_cfg)
                if base > 0:
                    return min(base + slice_last_frames, num_training_frames)
                else:
                    return num_training_frames

        kv_frames = _resolve_kv_frames(self.local_attn_size) # always 21 帧
        
        if DEBUG:
            print(f"[KV policy] local_attn_size={self.local_attn_size} slice_last_frames={slice_last_frames} num_training_frames={num_training_frames} -> kv_frames={kv_frames}")
        self.kv_cache_size = int(kv_frames) * self.frame_seq_length # 21 * 1125 = 23625

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            # Generate random indices
            indices = torch.randint( # [num_blocks, 1]
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            ) # 随机生成 num_blocks 个索引（范围是 0 到 num_denoising_steps-1），每个 block 一个
            if self.last_step_only: # false
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            slice_last_frames: int = 21,
            step: Optional[int] = None,
            visualization: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        
        batch_size, num_frames, num_channels, height, width = noise.shape
        actual_frame_seq_length = (height // 2) * (width // 2)
        
        assert num_frames % self.num_frame_per_block == 0 # 21 = 21 % 3 == 0
        num_blocks = num_frames // self.num_frame_per_block # 7 blocks
        
        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_ar_generate_cache()

        # Step 2: Cache context feature
        current_start_frame = 0

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks # [3,3,3,3,3,3,3]
        num_denoising_steps = len(self.denoising_step_list)
        self.num_denoising_steps = num_denoising_steps
        
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device) # 返回一个index # [7, 1]
        self.exit_flags_cache = exit_flags

        start_gradient_frame_index = 0
        gradient_index_list = [num_blocks-1]
        grad_enable_mask = torch.zeros((batch_size, sum(all_num_frames)), dtype=torch.bool)
        
        
        # Pad face pixel values to match the expected latent frames （4n+1）
        # pad_face_pixel_values = conditional_dict["face_pixel_values"][:, :, -1:, :, :].repeat(1, 1, 3, 1, 1) # 1 3 3 512 512
        # pad_face_pixel_values = pad_face_pixel_values.to(device=noise.device, dtype=noise.dtype)
        # conditional_dict["face_pixel_values"] = torch.cat([conditional_dict["face_pixel_values"], pad_face_pixel_values], dim=2)
        
        # if dist.get_rank() == 0:
        #     import pdb; pdb.set_trace()

        # cond prefill for each forward
        if conditional_dict.get('sink_flag', False):
            for index, current_timestep in enumerate(self.denoising_step_list):
                
                if dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, sink_flag True, in foward sink index:{index}, current_timestep: {current_timestep}")

                noisy_input = conditional_dict["ref_lat"] # 1 4 16 90 90
                
                block_cond = copy.deepcopy(conditional_dict)
                # B C f_lat h_lat w_lat
                block_cond['pose_latents'] = None
                block_cond["face_pixel_values"] = None
                block_cond["y"] = block_cond["y"][:, :, 0:1] # 1 20 1 90 90
                
                timestep = torch.ones(
                        [batch_size, self.num_frame_per_block],
                        device=noise.device,
                        dtype=torch.int64) * 0

                # if dist.get_rank() == 0:
                #     print(f"[before sink] index={index}, "
                #     f"cond_k id={id(self.kv_cache1[index][0]['cond_k'])}")
                #     f"kv_cache[0][0] id={id(self.kv_cache1[index][0])}"  # ← 加这行，看 dict 本身的 id

                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=block_cond,
                        timestep=timestep,
                        kv_cache=self.kv_cache1, # 解决fsdp直接cast和new一个新的kv cache
                        crossattn_cache=self.crossattn_cache,
                        current_start=0, # 0
                        is_sink=True
                    )

                # if dist.get_rank() == 0:
                #     print(f"[sink done] index={index}, "
                #     f"cond_k id={id(self.kv_cache1[index][0]['cond_k'])}")
                #     f"kv_cache[0][0] id={id(self.kv_cache1[index][0])}"  # ← 加这行，看 dict 本身的 id
                    # import pdb; pdb.set_trace()

        conditional_dict['sink_flag'] = False         
        conditional_dict["y"] = conditional_dict["y"][:, :, 1:, :, :] # remove the first frame for y, now y and pose_latents are aligned in time (num_frames)
            
        for block_index, current_num_frames in enumerate(all_num_frames):

            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]            
            block_cond = copy.deepcopy(conditional_dict)
            block_cond['pose_latents'] = block_cond['pose_latents'][:, :, current_start_frame:current_start_frame + current_num_frames]
            block_cond["face_pixel_values"] = block_cond["face_pixel_values"][:, :, current_start_frame*4:(current_start_frame + current_num_frames)*4]
            block_cond["y"] = block_cond["y"][:, :, current_start_frame:current_start_frame + current_num_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):

                if dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, sink_flag False, in foward inference index:{index}, current_timestep: {current_timestep}")
                
                # 表示当前 timestep 是否是本 block 需要反向传播的步
                if self.same_step_across_blocks: # same_step_across_blocks is true
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                
                if DEBUG and dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, current_start_frame: {current_start_frame}, current_num_frames: {current_num_frames}, current_timestep: {current_timestep},frame_seq_length: {self.frame_seq_length}")
                    print(f"self.kv_cache1,layer0,k: {self.kv_cache1[0]['k'].shape}")
                
                if not exit_flag: # no grad
                    if DEBUG and dist.get_rank() == 0:
                        print(f"not exit_flag,current_start_frame: {current_start_frame},timestep:{timestep}")
                    
                    with torch.no_grad():
                        if DEBUG and dist.get_rank() == 0:  
                            print(f"start denoising, frame{current_start_frame},timestep:{timestep}")

                        _, denoised_pred = self.generator( # from xt -> x0
                            noisy_image_or_video=noisy_input, # 1 3 16 90 90
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * actual_frame_seq_length
                        )
                        
                        if DEBUG and dist.get_rank() == 0:
                            print(f"end denoising, frame{current_start_frame},timestep:{timestep}")
                        
                        # from x0 -> xt-1
                        next_timestep = self.denoising_step_list[index + 1]
                        random_noise = torch.randn_like(denoised_pred.flatten(0, 1)) # 3 16 90 90
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            random_noise,
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2]) # 1 3 16 90 90
                
                
                else:  # exit_flag true，grad开启
                    self.noisy_input_cache[int(block_index)] = noisy_input
                    if (block_index not in gradient_index_list) or (current_start_frame < start_gradient_frame_index):
                        if DEBUG and dist.get_rank() == 0:
                            print(f"current_start_frame < start_gradient_frame_index,current_start_frame: {current_start_frame},start_gradient_frame_index: {start_gradient_frame_index},timestep:{timestep}")
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = False
                        with torch.no_grad():

                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=block_cond,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * actual_frame_seq_length
                            )
                    else:
                        if DEBUG and dist.get_rank() == 0:
                            print(f"current_start_frame >= start_gradient_frame_index,current_start_frame: {current_start_frame},start_gradient_frame_index: {start_gradient_frame_index},timestep:{timestep}")
                        # print(f"enable grad: {current_start_frame}")
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = True
 
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * actual_frame_seq_length
                        )

                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            if self.rerun_kv_cache:
                context_timestep = torch.ones_like(timestep) * self.context_noise 
                random_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    random_noise,
                    context_timestep * torch.ones(
                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
                
                if DEBUG and dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, current_start_frame: {current_start_frame}, current_num_frames: {current_num_frames}, current_timestep: {current_timestep}")
                    print(f"rank {dist.get_rank()}, rerun_for_cache")
                    
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=block_cond,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * actual_frame_seq_length 
                    )
            
            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if dist.get_rank() == 0 and DEBUG:
            print(f"grad_enable_mask: {grad_enable_mask[0, :]}")
            
        # Step 3.5: Return the denoised timestep
        denoising_step_list_tensor = torch.tensor(self.denoising_step_list, dtype=torch.long)
        if not self.same_step_across_blocks: # same_step_across_blocks is true
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - denoising_step_list_tensor[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - denoising_step_list_tensor[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - denoising_step_list_tensor[exit_flags[0]].cuda()).abs(), dim=0).item()

        self.output_cache = output.detach().clone() # accumulation_step为0的时候，缓存output用于下一次inference_with_trajectory_with_cache调用

        # Visualization logic
        rank = dist.get_rank() if dist.is_initialized() else 0
        if getattr(self, "vae", None) is not None:
            if visualization:
                # import pdb; pdb.set_trace()
                print(f"Visualizing output at rank {rank}...")
                with torch.no_grad():
                    # Decode latents to pixels
                    pixels = self.vae.decode_to_pixel(output) # B T C H W
                    # Convert to [T, C, H, W] uint8
                    video = ((pixels[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                    video = video.permute(0, 2, 3, 1) # T H W C

                    # Save video
                    from torchvision.io import write_video
                    save_dir = os.environ.get("VIS_SAVE_DIR", "visualizations")
                    os.makedirs(save_dir, exist_ok=True)
                    if step is not None:
                        file_path = os.path.join(save_dir, f"debug_output_step_{step:06d}_rank{rank}.mp4")
                    write_video(file_path, video.cpu(), fps=16)
                    print(f"Video saved to {file_path}")

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to
    
    def inference_with_trajectory_with_cache(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            slice_last_frames: int = 21,
            accumulation_step: int = 0,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        actual_frame_seq_length = (height // 2) * (width // 2)


        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_output_frames = num_frames
        output = self.output_cache.clone().detach()

        # Step 1: Should not initialize KV cache to all zeros

        # Step 2: Cache context feature
        current_start_frame = 0

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        exit_flags = self.exit_flags_cache
        
        # start_gradient_frame_index = num_output_frames - slice_last_frames
        start_gradient_frame_index = 0
        gradient_index_list = [num_blocks - 1 - int(accumulation_step)] # 根据accumulation_step选择num_blocks来计算梯度(5,4,...)

        grad_enable_mask = torch.zeros((batch_size, sum(all_num_frames)), dtype=torch.bool) # all false
        # If static local_attn_size, set it first
        if not isinstance(self.local_attn_size, (list, tuple)):
            self.generator.model.local_attn_size = int(self.local_attn_size) # 12
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        
        conditional_dict['sink_flag'] = False
        
        # Pad face pixel values to match the expected latent frames （4n+1）
        # 反过来 pad 呢？也许更好一些！
        pad_face_pixel_values = conditional_dict["face_pixel_values"][:, :, -1:, :, :].repeat(1, 1, 3, 1, 1) # 1 3 3 512 512
        pad_face_pixel_values = pad_face_pixel_values.to(device=noise.device, dtype=noise.dtype)
        conditional_dict["face_pixel_values"] = torch.cat([conditional_dict["face_pixel_values"], pad_face_pixel_values], dim=2)
        
        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            if block_index not in gradient_index_list: # 直接跳过不计算梯度的block
                current_start_frame += current_num_frames
                continue
            
            block_cond = copy.deepcopy(conditional_dict)
            block_cond['pose_latents'] = block_cond['pose_latents'][:, :, current_start_frame:current_start_frame + current_num_frames]
            block_cond["face_pixel_values"] = block_cond["face_pixel_values"][:, :, current_start_frame*4:(current_start_frame + current_num_frames)*4]
            block_cond["y"] = block_cond["y"][:, :, current_start_frame:current_start_frame + current_num_frames]
            
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                
                if self.same_step_across_blocks: # false
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                
                if not exit_flag:
                    continue
                else:
                    noisy_input = self.noisy_input_cache[int(block_index)].detach().clone()
                    if DEBUG and dist.get_rank() == 0:
                        print(f"current_start_frame >= start_gradient_frame_index,current_start_frame: {current_start_frame},start_gradient_frame_index: {start_gradient_frame_index},timestep:{timestep}")
                    # print(f"enable grad: {current_start_frame}")
                    grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = True

                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=block_cond,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * actual_frame_seq_length 
                    )

                    break

            # Step 3.2: record the model's output
            # output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            # # Step 3.2: build differentiable output by concatenation
            left  = self.output_cache[:, :current_start_frame].detach()  # 常量段
            mid   = denoised_pred                                        # 可导段
            right = self.output_cache[:, current_start_frame + current_num_frames:].detach()  # 常量段
            output = torch.cat([left, mid, right], dim=1)

            # Step 3.3: rerun with timestep zero to update the cache
            if self.rerun_kv_cache:
                context_timestep = torch.ones_like(timestep) * self.context_noise 
                random_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    random_noise,
                    context_timestep * torch.ones(
                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
                
                if DEBUG and dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, current_start_frame: {current_start_frame}, current_num_frames: {current_num_frames}, current_timestep: {current_timestep}")
                    print(f"rank {dist.get_rank()}, rerun_for_cache")
                    
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=block_cond,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * actual_frame_seq_length 
                    )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if dist.get_rank() == 0 and DEBUG:
            print(f"grad_enable_mask: {grad_enable_mask[0, :]}")
            
        # Step 3.5: Return the denoised timestep
        denoising_step_list_tensor = torch.tensor(self.denoising_step_list, dtype=torch.long)
        if not self.same_step_across_blocks: # same_step_across_blocks is true
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - denoising_step_list_tensor[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - denoising_step_list_tensor[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - denoising_step_list_tensor[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _free_cache_memory(self, accumulation_step: int, num_blocks_clip: int = 4):
        full_kv_cache_size = self.kv_cache_size
        keep_size = full_kv_cache_size // num_blocks_clip * (num_blocks_clip - 1 - accumulation_step)

        for blk in self.kv_cache1:
            blk["k"]     = blk["k"]    [:, :keep_size].clone().detach()
            if "k_raw" in blk:
                blk["k_raw"] = blk["k_raw"][:, :keep_size].clone().detach()
            blk["v"]     = blk["v"]    [:, :keep_size].clone().detach()
            blk["cond_k"] = blk["cond_k"].detach()
            blk["cond_v"] = blk["cond_v"].detach()
            blk["global_end_index"] = blk["global_end_index"].detach()
            blk["local_end_index"] = blk["local_end_index"].detach()

        torch.cuda.empty_cache()


    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        heads, dim_head = (40, 128) if BASE_MODEL == "14B" else (12, 128)
        if DEBUG:
            print(f"rank {dist.get_rank()} initialize kv cache with batch_size: {batch_size}, kv_cache_size: {self.kv_cache_size}")
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k":     torch.zeros([batch_size, self.kv_cache_size, heads, dim_head], dtype=dtype, device=device),
                "k_raw": torch.zeros([batch_size, self.kv_cache_size, heads, dim_head], dtype=dtype, device=device),
                "v":     torch.zeros([batch_size, self.kv_cache_size, heads, dim_head], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":  torch.tensor([0], dtype=torch.long, device=device),
                "cond_k": torch.zeros([batch_size, 3000, 40, 128], dtype=dtype, device=device),
                "cond_v": torch.zeros([batch_size, 3000, 40, 128], dtype=dtype, device=device),
                "cond_end": torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache1 = kv_cache1

    def _initialize_ar_generate_cache(self):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        self.noisy_input_cache = {} 
        self.output_cache = None
        self.exit_flags_cache = None

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []
        heads,dim_head = (0,128) if BASE_MODEL == "14B" else (12,128)
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 0, heads, dim_head], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 0, heads, dim_head], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
    
    def clear_kv_cache_memory(self):
        """
        Clear the memory of the KV cache.
        """
        self.kv_cache1 = None
        torch.cuda.empty_cache()  

    def clear_kv_cache(self):
        """
        Zero out all tensors in KV cache and cross-attention cache instead of setting them to None.
        This preserves memory allocation while clearing old information, avoiding reallocation overhead.
        """

        # Clear KV cache
        if getattr(self, "kv_cache1", None) is not None:
            for blk in self.kv_cache1:
                blk["k"].zero_()
                if "k_raw" in blk:
                    blk["k_raw"].zero_()
                blk["v"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()
                if "cond_k" in blk:
                    blk["cond_k"].zero_()
                if "cond_v" in blk:
                    blk["cond_v"].zero_()
                if "cond_end" in blk:
                    blk["cond_end"].zero_()

        # Clear cross-attention cache
        if getattr(self, "crossattn_cache", None) is not None:
            for blk in self.crossattn_cache:
                blk["k"].zero_()
                blk["v"].zero_()
                blk["is_init"] = False

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set a unified upper bound for all submodules that contain the max_attention_size attribute.
        local_attn_size_value == -1 indicates global attention (use Wan's default token limit 32760).
        Otherwise set to local_attn_size_value * frame_seq_length.
        """
        if isinstance(local_attn_size_value, (list, tuple)):
            raise ValueError("_set_all_modules_max_attention_size expects an int, got list/tuple.")

        if int(local_attn_size_value) == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        # Root module
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                _ = getattr(self.generator.model, "max_attention_size")
            except Exception:
                pass
            setattr(self.generator.model, "max_attention_size", target_size)

        # Child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", target_size)
                except Exception:
                    pass
