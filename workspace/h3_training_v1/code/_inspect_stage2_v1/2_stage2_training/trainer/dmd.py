# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import time
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.distributed as dist
from utils.memory import log_gpu_memory
from trainer.base import SelfForcingModel
from diffsynth.wan_wrapper import WanDiffusionWrapper
from utils.debug_option import DEBUG, LOG_GPU_MEMORY


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.min_num_training_frames = getattr(args, "min_num_training_frames", 21)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()
            # The teacher is frozen and its forwards run under no_grad, so it
            # keeps no backward graph -- but its per-layer activations are still
            # materialised. Checkpointing bounds that transient footprint, which
            # matters because this run holds three 14B models at once.
            self.real_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"): # False
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
        
        self.generator_output_cache = {}
        self.cache_generator_output = getattr(args, "cache_generator_output", False) # false

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        step=0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        fake_cond = dict(conditional_dict)
        fake_cond["y"] = conditional_dict.get("y_for_bid", conditional_dict["y"])
        fake_cond["pose_latents"] = conditional_dict.get("pose_latents_for_bid", conditional_dict["pose_latents"])
        fake_cond["face_pixel_values"] = conditional_dict.get("face_pixel_values_for_bid", conditional_dict["face_pixel_values"])

        fake_uncond = dict(unconditional_dict)
        fake_uncond["y"] = unconditional_dict.get("y_for_bid", unconditional_dict["y"])
        fake_uncond["pose_latents"] = unconditional_dict.get("pose_latents_for_bid", unconditional_dict["pose_latents"])
        fake_uncond["face_pixel_values"] = unconditional_dict.get("face_pixel_values_for_bid", unconditional_dict["face_pixel_values"])
        
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=fake_cond,
            timestep=timestep
        )
        if self.fake_guidance_scale != 0.0: # default 0.0
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=fake_uncond,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond
            

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        # teacher uses bidirectional conditioning: y_for_bid / pose_latents_for_bid / face_pixel_values_for_bid
        real_cond = dict(conditional_dict)
        real_cond["y"] = conditional_dict.get("y_for_bid", conditional_dict["y"])
        real_cond["pose_latents"] = conditional_dict.get("pose_latents_for_bid", conditional_dict["pose_latents"])
        real_cond["face_pixel_values"] = conditional_dict.get("face_pixel_values_for_bid", conditional_dict["face_pixel_values"])

        real_uncond = dict(unconditional_dict)
        real_uncond["y"] = unconditional_dict.get("y_for_bid", unconditional_dict["y"])
        real_uncond["pose_latents"] = unconditional_dict.get("pose_latents_for_bid", unconditional_dict["pose_latents"])
        real_uncond["face_pixel_values"] = unconditional_dict.get("face_pixel_values_for_bid", unconditional_dict["face_pixel_values"])

        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=real_cond,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=real_uncond,
            timestep=timestep
        )

        # cfg
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # Same interval knob as the trainer; and unlike upstream this is not
        # re-entered once per accumulation step, which used to overwrite the
        # same mp4 four times per visualised step on every rank.
        if step % getattr(self.args, "vis_interval", 50) == 0:
            # Visualization logic
            rank = dist.get_rank() if dist.is_initialized() else 0
            if getattr(self, "vae", None) is not None:
                # import pdb; pdb.set_trace()
                print(f"Visualizing output at rank {rank}...")
                with torch.no_grad():
                    # Decode latents to pixels
                    pixels = self.vae.decode_to_pixel(pred_real_image) # B T C H W
                    # Convert to [T, C, H, W] uint8
                    video = ((pixels[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                    video = video.permute(0, 2, 3, 1) # T H W C

                    # Save video
                    # torchvision.io.write_video needs PyAV, which is not installed
                    # and whose wheel may not carry libx264. This tree already ships
                    # an imageio-ffmpeg writer that stage1 has used for days.
                    from diffsynth.utils.data import save_video as _save_video
                    save_dir = os.environ.get("VIS_SAVE_DIR") + "real"
                    os.makedirs(save_dir, exist_ok=True)
                    if step is not None:
                        file_path = os.path.join(save_dir, f"debug_output_step_{step:06d}_rank{rank}.mp4")
                    _save_video(video.cpu().numpy(), file_path, fps=16, quality=5)
                    print(f"Video saved to {file_path}")

            # Visualization logic
            rank = dist.get_rank() if dist.is_initialized() else 0
            if getattr(self, "vae", None) is not None:
                # import pdb; pdb.set_trace()
                print(f"Visualizing output at rank {rank}...")
                with torch.no_grad():
                    # Decode latents to pixels
                    pixels = self.vae.decode_to_pixel(pred_fake_image) # B T C H W
                    # Convert to [T, C, H, W] uint8
                    video = ((pixels[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                    video = video.permute(0, 2, 3, 1) # T H W C

                    # Save video
                    # torchvision.io.write_video needs PyAV, which is not installed
                    # and whose wheel may not carry libx264. This tree already ships
                    # an imageio-ffmpeg writer that stage1 has used for days.
                    from diffsynth.utils.data import save_video as _save_video
                    save_dir = os.environ.get("VIS_SAVE_DIR") + "fake"
                    os.makedirs(save_dir, exist_ok=True)
                    if step is not None:
                        file_path = os.path.join(save_dir, f"debug_output_step_{step:06d}_rank{rank}.mp4")
                    _save_video(video.cpu().numpy(), file_path, fps=16, quality=5)
                    print(f"Video saved to {file_path}")

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video[:, -pred_real_image.shape[1]:] - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video # B T C H W
        batch_size, num_frame = image_or_video.shape[:2]

        # if dist.get_rank() == 0:
        #     import pdb; pdb.set_trace()

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep, # 0
                max_timestep, # 1000
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            timestep = timestep.to(torch.float32)
            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # add clean ref image
            noisy_latent = torch.cat([conditional_dict["ref_lat"], noisy_latent], dim=1) # B T+1 C H W

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                step=step
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        accumulation_step: int = 0,
        step: Optional[int] = None,
        visualization: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        
        # Step 1: Unroll generator to obtain fake videos
        slice_last_frames = getattr(self.args, "slice_last_frames", 21) # 21
        _t_gen_start = time.time()
        if DEBUG and dist.get_rank() == 0:
            print(f"generator_rollout")
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames,
            accumulation_step=accumulation_step,
            step=step,
            visualization=visualization
        )
        if self.cache_generator_output: # false
            self.generator_output_cache['pred_image'] = pred_image.detach().clone()
            self.generator_output_cache['denoised_timestep_from'] = denoised_timestep_from
            self.generator_output_cache['denoised_timestep_to'] = denoised_timestep_to
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {pred_image.shape},dtype: {pred_image.dtype}")
            print(f"cond_states in cond: {conditional_dict['cond_states'].shape}")
            print(f"ref latents in cond: {conditional_dict['ref_latents'].shape}")
            print(f"cond_states in uncond: {unconditional_dict['cond_states'].shape}")
            print(f"ref latents in uncond: {unconditional_dict['ref_latents'].shape}")
            if gradient_mask is not None:   
                print(f"gradient_mask: {gradient_mask[0, :, 0, 0, 0]}")
            else:
                print(f"gradient_mask: None")
        
        gen_time = time.time() - _t_gen_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After generator unroll", device=self.device, rank=dist.get_rank())
        
        # Step 2: Compute the DMD loss
        # y_for_bid / pose_latents_for_bid / face_pixel_values_for_bid are passed
        # directly to real_score inside _compute_kl_grad; no need to overwrite y here.
        
        _t_loss_start = time.time()
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image, # the generated image/video from the generator()
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            step=step
        )
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After compute_distribution_matching_loss", device=self.device, rank=dist.get_rank())

        loss_time = time.time() - _t_loss_start
        # print(f"[GeneratorLoss] loss {loss_val} | gen_time {gen_time:.3f}s | loss_time {loss_time:.3f}s")

        dmd_log_dict.update({
            "gen_time": gen_time,
            "loss_time": loss_time
        })

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        step: Optional[int] = None,
        visualization: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # Step 1: Run generator on backward simulated noisy input
        _t_gen_start = time.time()

        if self.cache_generator_output:
            if DEBUG and dist.get_rank() == 0:
                print(f"use cached generator output")
            generated_image = self.generator_output_cache['pred_image'].clone()
            denoised_timestep_from = self.generator_output_cache['denoised_timestep_from']
            denoised_timestep_to = self.generator_output_cache['denoised_timestep_to']
        else:
            with torch.no_grad():
                if DEBUG and dist.get_rank() == 0:
                    print(f"critic_rollout")
                generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent,
                    slice_last_frames=slice_last_frames,
                    step=step,
                    visualization=visualization
                )
                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(f"Critic loss: After run_generator", device=self.device, rank=dist.get_rank())
                self.inference_pipeline.clear_kv_cache_memory()
                if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                    log_gpu_memory(f"Critic loss: After clear_kv_cache", device=self.device, rank=dist.get_rank())
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {generated_image.shape}")

        # if dist.get_rank() == 0:
        #     import pdb; pdb.set_trace()

        gen_time = time.time() - _t_gen_start
        batch_size, num_frame = generated_image.shape[:2]

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After generator unroll", device=self.device, rank=dist.get_rank())
        _t_loss_start = time.time()

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frame, # add a reference frame
            self.num_frame_per_block,
            uniform_timestep=True
        )
        if dist.get_rank() == 0:
            print("critic_timestep:", critic_timestep)

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)
        critic_timestep = critic_timestep.to(torch.float32)

        if dist.get_rank() == 0:
            print("After shift critic_timestep:", critic_timestep)

        critic_noise = torch.randn_like(generated_image) # 1 22 16 50 90

        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        # add ref image
        noisy_generated_image = torch.cat([conditional_dict["ref_lat"], noisy_generated_image], dim=1)

        fake_cond = dict(conditional_dict)
        fake_cond["y"] = conditional_dict.get("y_for_bid", conditional_dict["y"])
        fake_cond["pose_latents"] = conditional_dict.get("pose_latents_for_bid", conditional_dict["pose_latents"])
        fake_cond["face_pixel_values"] = conditional_dict.get("face_pixel_values_for_bid", conditional_dict["face_pixel_values"])

        fake_uncond = dict(unconditional_dict)
        fake_uncond["y"] = unconditional_dict.get("y_for_bid", unconditional_dict["y"])
        fake_uncond["pose_latents"] = unconditional_dict.get("pose_latents_for_bid", unconditional_dict["pose_latents"])
        fake_uncond["face_pixel_values"] = unconditional_dict.get("face_pixel_values_for_bid", unconditional_dict["face_pixel_values"])


        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=fake_cond,
            timestep=critic_timestep
        )

        # pred_fake_image 是 content only，xt 也只取 content 部分（去掉 ref 帧）
        noisy_content = noisy_generated_image[:, 1:]
        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_content.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_content.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        loss_time = time.time() - _t_loss_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After denoising loss", device=self.device, rank=dist.get_rank())

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            "gen_time": gen_time,
            "loss_time": loss_time
        }

        return denoising_loss, critic_log_dict
