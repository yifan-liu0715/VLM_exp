# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from typing import Tuple, Optional
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.self_forcing_training import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from diffsynth.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanImageEncoderWrapper

class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = args.denoising_step_list # !!!!

            if args.warp_denoising_step: # false
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.local_attn_size = getattr(args, "model_kwargs", {}).get("local_attn_size", -1)

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        # Upstream hardcodes a bidirectional teacher and loads a separate
        # `bid_lora_ckpt` -- a bidirectional stage1 run this project never did.
        # The paper describes stage2 differently: "The teacher-forcing adaptation
        # produces a high-quality causal generator, but it still requires 50
        # denoising steps per block", i.e. stage1's own causal model IS the
        # teacher and stage2 distils it from 50 steps down to 3. Under that
        # reading teacher and student share an architecture and an init, and the
        # distillation signal is the step budget, not the attention pattern.
        # `teacher_is_causal: true` follows the paper; false reproduces upstream
        # and then bid_lora_ckpt must point at a real bidirectional stage1 LoRA.
        teacher_is_causal = bool(getattr(args, "teacher_is_causal", False))
        teacher_kwargs = dict(getattr(args, "model_kwargs", {})) if teacher_is_causal else {}
        self.real_score = WanDiffusionWrapper(**teacher_kwargs, is_causal=teacher_is_causal)
        self.real_score.model.requires_grad_(False)
        if getattr(args, "num_frame_per_block", 1) > 1 and teacher_is_causal:
            self.real_score.model.num_frame_per_block = args.num_frame_per_block

        # self.fake_score = WanDiffusionWrapper(is_causal=False)
        # fake_score 与 real_score 结构和权重完全相同，直接 deepcopy 跳过第三次磁盘读取
        import copy
        self.fake_score = copy.deepcopy(self.real_score)
        self.fake_score.model.requires_grad_(True)

        # 三个模型都加载完，释放 DIT state_dict 缓存，节省 CPU 内存
        import diffsynth.wan_wrapper as _ww
        _ww._cached_dit_state_dict = None

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        
        self.image_encoder = WanImageEncoderWrapper()
        self.image_encoder.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            # if is_sequence_parallel_initialized():
            #     timestep = broadcast_tensor(timestep, src=0)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )

            # make the noise level the same within every block
            if self.independent_first_frame: # true
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:] # [B, F-1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block) # [B, (F-1)/num_frame_per_block, num_frame_per_block]
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1] # make same within block, except the first frame of each block [B, (F-1)/num_frame_per_block, num_frame_per_block]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1) # [B, F-1]
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)() # FlowPredLoss

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        slice_last_frames: int = 21,
        accumulation_step: int = 0,
        step: Optional[int] = None,
        visualization: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent

        noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [min_num_frames, self.num_training_frames], but still being a multiple of self.num_frame_per_block.
        # If `min_num_frames` is not provided, we fallback to the original default behaviour.
        min_num_frames = self.min_num_training_frames
        max_num_frames = self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        
        max_num_blocks = max_num_frames // self.num_frame_per_block # 21 // 3 = 7
        min_num_blocks = min_num_frames // self.num_frame_per_block # 21 // 3 = 7
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device) # 21
        
        # Sync num_generated_blocks across all processes
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if dist.get_rank() == 0:
            print(f"num_generated_frames: {num_generated_frames}")

        noise_shape[1] = num_generated_frames
        noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype) # b c f h w

        # forward simulation to get the generator's input
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=noise, # 1 x f h w
            slice_last_frames=slice_last_frames, # 21
            accumulation_step=accumulation_step, # 0 - 6
            step=step,
            visualization=visualization,
            **conditional_dict
        )
        
        # Decide whether to slice based on `slice_last_frames`; when `slice_last_frames == -1`, keep all frames
        if slice_last_frames != -1 and pred_image_or_video.shape[1] > slice_last_frames:
            with torch.no_grad():
                # Re-encode: take all frames before the last (slice_last_frames - 1) frames for pixel decoding
                if slice_last_frames > 1:
                    latent_to_decode = pred_image_or_video[:, :-(slice_last_frames - 1), ...]
                else:
                    latent_to_decode = pred_image_or_video
                # Decode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            if slice_last_frames > 1:
                last_frames = pred_image_or_video[:, -(slice_last_frames - 1):, ...]
                pred_image_or_video_sliced = torch.cat([image_latent, last_frames], dim=1)
            else:
                pred_image_or_video_sliced = image_latent
        else:
            pred_image_or_video_sliced = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_sliced, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_sliced = pred_image_or_video_sliced.to(self.dtype)
        return pred_image_or_video_sliced, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        slice_last_frames: int = 21,
        accumulation_step: int = 0,
        step: Optional[int] = None,
        visualization: bool = False,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        if accumulation_step == 0: # 就跑一次no grad并缓存下latent, 并反向传播一下第-1个block的gradient
            return self.inference_pipeline.inference_with_trajectory(
                noise=noise, **conditional_dict, slice_last_frames=slice_last_frames, step=step, visualization=visualization
            )
        else: # 其他的就跑第i个block的gradient计算（利用缓存下来的kv cache）
            return self.inference_pipeline.inference_with_trajectory_with_cache(
                noise=noise, **conditional_dict, slice_last_frames=slice_last_frames, accumulation_step=accumulation_step
            )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        local_attn_size = getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1)
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # do not use self.num_training_frames, because it is changed by generator_loss and critic_loss
        num_training_frames = getattr(self.args, "num_training_frames")
        # 把 bucket list 传给 pipeline，用于动态计算 frame_seq_length
        target_hw = getattr(self.args, "v2v_training_config", {}).get("dataset", {}).get("target_hw", None)
        target_hw_buckets = None
        if target_hw is not None:
            try:
                # OmegaConf ListConfig 也能 iterate，统一转成 python list
                hw_list = list(target_hw)
                if len(hw_list) > 0:
                    first = hw_list[0]
                    # 判断是否是 bucket list（每个元素是 [h, w]）
                    if hasattr(first, '__iter__') and not isinstance(first, str):
                        target_hw_buckets = [(int(h), int(w)) for h, w in hw_list]
            except Exception:
                pass
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=num_training_frames,
            rerun_kv_cache=self.args.rerun_kv_cache,
            context_noise=self.args.context_noise,
            local_attn_size=local_attn_size,
            slice_last_frames=slice_last_frames,
            num_training_frames=num_training_frames,
            vae=self.vae,
            target_hw_buckets=target_hw_buckets,
        )
