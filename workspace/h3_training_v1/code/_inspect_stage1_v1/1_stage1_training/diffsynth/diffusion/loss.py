from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss

def FlowMatchInContextSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    ref_frame = inputs["input_latents"][:, :, :1]  # 1 16 1 90 90
    content_sections = inputs["input_latents"][:, :, 1:]  # 1 16 13 90 90
    noise = torch.randn_like(content_sections) # 1 16 13 90 90
    noisy_content_sections = pipe.scheduler.add_noise(content_sections, noise, timestep) # 1 16 13 90 90
    training_target = pipe.scheduler.training_target(content_sections, noise, timestep) # 1 16 13 90 90
    
    inputs["latents"] = torch.cat([ref_frame, noisy_content_sections], dim=2) # 1 16 14 90 90

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep) # 1 16 14 90 90
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float()) # 1 16 13 90 90
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchInContextMotionSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    cond_len = inputs["cond_lens"]
    motion_timestep = inputs["motion_timestep"]
    ref_frame = inputs["input_latents"][:, :, :1]  # 1 16 1 90 90
    motion_frame = inputs["input_latents"][:, :, 1:cond_len]  # 1 16 3 90 90
    content_sections = inputs["input_latents"][:, :, cond_len:]  # 1 16 13 90 90
    
    noise = torch.randn_like(content_sections) # 1 16 13 90 90
    noisy_content_sections = pipe.scheduler.add_noise(content_sections, noise, timestep) # 1 16 13 90 90
    training_target = pipe.scheduler.training_target(content_sections, noise, timestep) # 1 16 13 90 90
    
    noise_motion = torch.randn_like(motion_frame) # 1 16 3 90 90
    noisy_motion_frame = pipe.scheduler.add_noise(motion_frame, noise_motion, motion_timestep) # 1 16 3 90 90
    
    inputs["latents"] = torch.cat([ref_frame, noisy_motion_frame, noisy_content_sections], dim=2) # 1 16 14 90 90
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep) # 1 16 14 90 90
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float()) # 1 16 13 90 90
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss

def FlowMatchInContextDiffusionForcingLoss(pipe: BasePipeline, num_frames_per_block=3, **inputs):
    """
    Diffusion Forcing (Block-level):
    - ref_frame (block 0): 不加噪，作为条件帧
    - content_sections: 按 block 独立采样 timestep，block 内所有帧共享同一噪声级别
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    cond_lens = inputs["cond_lens"]
    ref_frame = inputs["input_latents"][:, :, :1]        # B 16 1 H W
    # motion_frame = inputs["input_latents"][:, :, 1:cond_lens]  # B 16 M H W
    content_sections = inputs["input_latents"][:, :, cond_lens:] # B 16 T H W
    B, C, T, H, W = content_sections.shape
        
    # # 对 motion_frame 采样一个独立的 timestep，并加噪（与 content_sections 的 timestep 不同）
    # motion_timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    # motion_timestep = pipe.scheduler.timesteps[motion_timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    # noise_motion = torch.randn_like(motion_frame) # 1 16 3 90 90
    # noisy_motion_frame = pipe.scheduler.add_noise(motion_frame, noise_motion, motion_timestep) # 1 16 3 90 90

    # noisy_motion_frame = motion_frame # 去掉motion的加躁试试？

    # 对 content_sections 按 block 采样 timestep，并加噪
    # ---- Block 数量 ----
    num_blocks = (T + num_frames_per_block - 1) // num_frames_per_block  # ceil
    # ---- 每个 block 独立采样一个 timestep_id ----
    block_timestep_ids = torch.randint(min_timestep_boundary, max_timestep_boundary, (B, num_blocks))  # (B, num_blocks)
    # ---- block-level timestep -> frame-level timestep (block 内重复) ----
    # (B, num_blocks) -> (B, num_blocks * num_frames_per_block) -> 裁剪到 T
    frame_timestep_ids = block_timestep_ids.repeat_interleave(num_frames_per_block, dim=1)[:, :T]  # (B, T)
    frame_timesteps = pipe.scheduler.timesteps[frame_timestep_ids].to(dtype=pipe.torch_dtype, device=pipe.device)  # (B, T)
    # frame_timesteps 原来是 (B,T)；按 block 扩展
    block_timesteps = pipe.scheduler.timesteps[block_timestep_ids].to(dtype=pipe.torch_dtype, device=pipe.device)  # (B, num_blocks)
    noise_content = torch.randn_like(content_sections) # B,16,T,H,W
    noisy_content = []
    for i in range(num_blocks):
        t = block_timesteps[:, i]
        noisy_content.append(pipe.scheduler.add_noise(
            content_sections[:, :, i*num_frames_per_block:(i+1)*num_frames_per_block],
            noise_content[:, :, i*num_frames_per_block:(i+1)*num_frames_per_block], t)
        )
    noisy_content = torch.cat(noisy_content, dim=2)

    # # 直接sft试试？
    # content_timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    # content_timestep = pipe.scheduler.timesteps[content_timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    # noise_content = torch.randn_like(content_sections) # 1 16 3 90 90
    # noisy_content = pipe.scheduler.add_noise(content_sections, noise_content, content_timestep) # 1 16 3 90 90
    
    # training_target 这里本身不依赖 timestep，可以直接用 vectorized 写法
    training_target = pipe.scheduler.training_target(content_sections, noise_content, timestep=None)

    # ---- 拼接 ref_frame 送入模型 ----
    inputs["latents"] = torch.cat([ref_frame, noisy_content], dim=2)  # B 16 T+1 H W
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=frame_timesteps)  # B 16 T+1 H W
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())

    return loss


def FlowMatchInContextTimeStepForcingLoss(pipe: BasePipeline, num_frames_per_block=3, **inputs):
    """
    TimeStepForcingLoss (Block-level):
    - ref_frame (block 0): 不加噪，作为条件帧
    - content_sections: 按 block 独立采样 timestep，block 内所有帧共享同一噪声级别
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    cond_lens = inputs["cond_lens"]
    ref_frame = inputs["input_latents"][:, :, :1]        # B 16 1 H W
    # motion_frame = inputs["input_latents"][:, :, 1:cond_lens]  # B 16 M H W
    content_sections = inputs["input_latents"][:, :, cond_lens:] # B 16 T H W
    B, C, T, H, W = content_sections.shape
        
    # # 对 motion_frame 采样一个独立的 timestep，并加噪（与 content_sections 的 timestep 不同）
    # motion_timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    # motion_timestep = pipe.scheduler.timesteps[motion_timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    # noise_motion = torch.randn_like(motion_frame) # 1 16 3 90 90
    # noisy_motion_frame = pipe.scheduler.add_noise(motion_frame, noise_motion, motion_timestep) # 1 16 3 90 90

    # noisy_motion_frame = motion_frame # 去掉motion的加躁试试？

    # # 对 content_sections 按 block 采样 timestep，并加噪
    # # ---- Block 数量 ----
    # num_blocks = (T + num_frames_per_block - 1) // num_frames_per_block  # ceil
    # # ---- 每个 block 独立采样一个 timestep_id ----
    # block_timestep_ids = torch.randint(min_timestep_boundary, max_timestep_boundary, (B, num_blocks))  # (B, num_blocks)
    # # ---- block-level timestep -> frame-level timestep (block 内重复) ----
    # # (B, num_blocks) -> (B, num_blocks * num_frames_per_block) -> 裁剪到 T
    # frame_timestep_ids = block_timestep_ids.repeat_interleave(num_frames_per_block, dim=1)[:, :T]  # (B, T)
    # frame_timesteps = pipe.scheduler.timesteps[frame_timestep_ids].to(dtype=pipe.torch_dtype, device=pipe.device)  # (B, T)
    # # frame_timesteps 原来是 (B,T)；按 block 扩展
    # block_timesteps = pipe.scheduler.timesteps[block_timestep_ids].to(dtype=pipe.torch_dtype, device=pipe.device)  # (B, num_blocks)
    # noise_content = torch.randn_like(content_sections) # B,16,T,H,W
    # noisy_content = []
    # for i in range(num_blocks):
    #     t = block_timesteps[:, i]
    #     noisy_content.append(pipe.scheduler.add_noise(
    #         content_sections[:, :, i*num_frames_per_block:(i+1)*num_frames_per_block],
    #         noise_content[:, :, i*num_frames_per_block:(i+1)*num_frames_per_block], t)
    #     )
    # noisy_content = torch.cat(noisy_content, dim=2)

    # 直接sft试试？
    content_timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    content_timestep = pipe.scheduler.timesteps[content_timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    noise_content = torch.randn_like(content_sections) # 1 16 3 90 90
    noisy_content = pipe.scheduler.add_noise(content_sections, noise_content, content_timestep) # 1 16 3 90 90
    
    # training_target 这里本身不依赖 timestep，可以直接用 vectorized 写法
    training_target = pipe.scheduler.training_target(content_sections, noise_content, timestep=None)

    # ---- 拼接 ref_frame 送入模型 ----
    inputs["latents"] = torch.cat([ref_frame, noisy_content], dim=2)  # B 16 T+1 H W
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=content_timestep)  # B 16 T+1 H W
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())

    return loss


def FlowMatchInContextTeacherForcingLoss(pipe: BasePipeline, **inputs):
    """
    I2V Teacher Forcing:
    - reference frames kept as condition
    - noisy content frames as x
    - clean_x provides teacher stream and uses TF attention mask in model
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    cond_lens = int(inputs.get("cond_lens", 1))
    cond_frames = inputs["input_latents"][:, :, :cond_lens]
    content_frames = inputs["input_latents"][:, :, cond_lens:]
    B, C, T, H, W = content_frames.shape

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))  # 1
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)  # 1
    frame_timesteps = timestep.repeat(B, T)  # (B, T)
    
    noise = torch.randn_like(content_frames)
    noisy_content = pipe.scheduler.add_noise(content_frames, noise, timestep)
    training_target = pipe.scheduler.training_target(content_frames, noise, timestep)

    clean_content = content_frames
    aug_t = None

    sync_noise_to_clean = int(inputs.get("tf_sync_noise_to_clean", 0))
    if sync_noise_to_clean == 1:
        clean_content = pipe.scheduler.add_noise(content_frames, noise, timestep)
        aug_t = timestep
    else:
        clean_aug_max = int(inputs.get("tf_clean_aug_max_timestep", 0))
        if clean_aug_max > 0:
            clean_aug_max = min(clean_aug_max, len(pipe.scheduler.timesteps) - 1)
            aug_t_id = torch.randint(clean_aug_max, len(pipe.scheduler.timesteps), (1,))
            aug_t = pipe.scheduler.timesteps[aug_t_id].to(dtype=pipe.torch_dtype, device=pipe.device)
            clean_noise = torch.randn_like(content_frames)
            clean_content = pipe.scheduler.add_noise(content_frames, clean_noise, aug_t)

    inputs["latents"] = torch.cat([cond_frames, noisy_content], dim=2)
    clean_x = torch.cat([cond_frames, clean_content], dim=2)

    # import torch.distributed as dist
    # if dist.get_rank() == 0:
    #     import pdb; pdb.set_trace()
        
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(
        **models,
        **inputs,
        timestep=frame_timesteps,
        clean_x=clean_x,
        aug_t=aug_t,
    )

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss

def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
