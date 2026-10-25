import torch, types
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from transformers import Wav2Vec2Processor

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..core.loader.file import load_state_dict

from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit_causal_animate_motion import CausalWanModel, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1, 
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: CausalWanModel = None
        self.dit2: CausalWanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = ("dit",)
        self.in_iteration_models_2 = ("dit2",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbed_User(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateFacePixelValues(),
        ]
        self.post_units = []
        
        self.model_fn = model_fn_wan_video

    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        model_paths: list[str] = None,
        tokenizer_path: str = None,
        device: Union[str, torch.device] = "cuda",
    ):
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_paths = model_paths
        tokenizer_path = tokenizer_path
        
        # initialize tokenizer
        tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean='whitespace')
        pipe.tokenizer = tokenizer
        
        # Text encoder
        text_encoder_state_dict = load_state_dict(model_paths[1], torch_dtype=torch.bfloat16, device=device)
        text_encoder_model_instance = WanTextEncoder()
        text_encoder_model_instance.load_state_dict(text_encoder_state_dict, strict=True, assign=True)
        pipe.text_encoder = text_encoder_model_instance

        # Image encoder
        clip_state_dict = load_state_dict(model_paths[3], torch_dtype=torch.bfloat16, device=device)
        fixed_state = {}
        for k, v in clip_state_dict.items():
            new_k = k if k.startswith("model.") else "model." + k
            fixed_state[new_k] = v
        filtered = {k: v for k, v in fixed_state.items() if k.startswith("model.visual.") or k == "model.log_scale"}
        clip_model_instance = WanImageEncoder()
        clip_model_instance.load_state_dict(filtered, strict=True, assign=True)
        pipe.image_encoder = clip_model_instance

        # DIT
        kwargs = {
                "dim": 5120,
                "eps": 1e-06,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "in_dim": 36,
                "num_heads": 40,
                "num_layers": 40,
                "out_dim": 16,
                "patch_size": [1, 2, 2],
                "text_dim": 4096,
                "has_image_input": True
            }
        state_dict = load_state_dict(file_path=model_paths[0], torch_dtype=torch.bfloat16, device=device)
        model = CausalWanModel(**kwargs)
        # A plain Wan 14B I2V checkpoint carries none of the Animate-only modules.
        # Let exactly those be missing and initialize them here; anything else
        # missing means the checkpoint does not match `kwargs`.
        animate_only = ("pose_patch_embedding", "motion_encoder", "face_adapter", "face_encoder")
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        unexplained = [k for k in missing if not k.startswith(animate_only)]
        if unexplained or unexpected:
            raise RuntimeError(
                f"DiT checkpoint mismatch. missing={unexplained[:8]} unexpected={list(unexpected)[:8]}"
            )
        if missing:
            init_from_scratch = sorted({k.split(".")[0] for k in missing})
            for name in init_from_scratch:
                getattr(model, name).to(dtype=torch_dtype, device=device)
            # The pose branch is additive (see CausalWanModel.after_patch_embedding),
            # so zero-init keeps the pretrained model untouched at step 0 and lets
            # the control signal come up gradually.
            if "pose_patch_embedding" in init_from_scratch:
                torch.nn.init.zeros_(model.pose_patch_embedding.weight)
                torch.nn.init.zeros_(model.pose_patch_embedding.bias)
            print(f"[dit] initialized from scratch (not in checkpoint): {init_from_scratch}")
        pipe.dit = model

        # vae
        vae_state_dict = load_state_dict(model_paths[2], torch_dtype=torch.bfloat16, device=device)
        fixed_state = {}
        for k, v in vae_state_dict.items():
            new_k = k if k.startswith("model.") else "model." + k
            fixed_state[new_k] = v
        vae_model_instance = WanVideoVAE()
        vae_model_instance.load_state_dict(fixed_state, strict=True, assign=True)
        pipe.vae = vae_model_instance
        
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        motion_video: Optional[list[Image.Image]] = None,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
        start_frame: Optional[int] = 0,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video, "motion_video": motion_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, start_frame=start_frame)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, start_frame=start_frame)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            ref_latents = inputs_shared["latents"][:, :, 0:1]  # 1 cond latents
            content_latents = inputs_shared["latents"][:, :, 1:]  # Other frames latents
            content_latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], content_latents) # only denoise content latents
            inputs_shared["latents"] = torch.concat([ref_latents, content_latents], dim=2) # Combine first frame latents back
        
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        
        # Decode
        # inputs_shared["latents"] = inputs_shared["latents"][:, :, 5:] # Decode时不需要去掉前5帧条件latents!!!
        
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "motion_video", "noise", "tiled", "tile_size", "tile_stride", "input_image", "height", "width"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, motion_video, noise, tiled, tile_size, tile_stride, input_image, height, width):
        if input_video is None:
            if input_image is not None: # (1, F, C, H, W) (1, C, F, H, W) 
                if isinstance(input_image, Image.Image):
                    input_image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
                ref_latents = pipe.vae.encode(input_image.transpose(0,1).unsqueeze(0), device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)

            noise = noise[:,:,1:] # 第一帧去掉！！
            noise = torch.concat([ref_latents, noise], dim=2)
            return {"latents": noise}
        
        pipe.load_models_to_device(self.onload_model_names)
        
        if isinstance(input_video, list):
            input_video = pipe.preprocess_video(input_video)
        else:
            input_video = rearrange(input_video, "b t c h w -> b c t h w")

        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = input_latents[:,:,1:] # 第一帧去掉！！
        if input_image is not None: # (1, F, C, H, W) (1, C, F, H, W) 
            input_image_expanded = input_image.transpose(0,1).unsqueeze(0)
            input_image_latents = pipe.vae.encode(input_image_expanded, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([input_image_latents, input_latents], dim=2)    
            cond_lens = 1
            
        noise = noise[:,:,1:]
        latents = torch.concat([input_image_latents, noise], dim=2)
        return {"latents": latents, "input_latents": input_latents, "cond_lens": cond_lens}


class WanVideoUnit_PromptEmbed_User(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt_emb": "prompt_emb", "positive": "positive"},
            input_params_nega={"prompt_emb": "neg_prompt_emb", "positive": "positive"},
            output_params=("context",),
        )
    
    def use_prompt_emb(self, pipe: WanVideoPipeline, prompt_emb):
        prompt_emb = prompt_emb.to(pipe.device, dtype=pipe.torch_dtype)
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt_emb, positive) -> dict:
        if prompt_emb is None:
            return {}
        prompt_emb = self.use_prompt_emb(pipe, prompt_emb)
        return {"context": prompt_emb}

class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive", "context": "context"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive", "context": "context"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive, context=None) -> dict:
        if prompt is None or context is not None:
            return {} if context is None else {"context": context}
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        
        if isinstance(input_image, Image.Image):
            input_image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        
        clip_context = pipe.image_encoder.encode_image([input_image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    

class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, num_frames, height, width, tiled, tile_size, tile_stride):
        padding_ref = torch.zeros(3, 1, height, width).to(pipe.device)
        padding_content = torch.zeros(3, num_frames, height, width).to(pipe.device) # content frames 为 num_frames 帧
        video_lat = pipe.vae.encode([padding_content.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        video_lat = video_lat[:, 1:].to(dtype=pipe.torch_dtype, device=pipe.device) # 第一帧去掉！！
        ref_lat = pipe.vae.encode([padding_ref.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]

        y = torch.cat([ref_lat, video_lat], dim=1)
        _,T,H,W = y.shape
        msk = torch.zeros(4, T, H, W, device=pipe.device, dtype=pipe.torch_dtype)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            output_params=("pose_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(animate_pose_video, list):
            animate_pose_video = pipe.preprocess_video(animate_pose_video)
        else:
            animate_pose_video = rearrange(animate_pose_video, "b t c h w -> b c t h w")
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        pose_latents = pose_latents[:, :, 1:] # 第一帧去掉！！
        return {"pose_latents": pose_latents}


class WanVideoUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("animate_face_video",),
            output_params=("face_pixel_values"),
        )

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        
        if isinstance(inputs_shared["animate_face_video"], list):
            inputs_shared["animate_face_video"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
            inputs_posi["face_pixel_values"] = inputs_shared["animate_face_video"]
        else:
            inputs_posi["face_pixel_values"] = rearrange(inputs_shared["animate_face_video"], "b t c h w -> b c t h w")
        inputs_posi["face_pixel_values"] = inputs_posi["face_pixel_values"][:,:,1:] # 第一帧去掉！！
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


def model_fn_wan_video(
    dit: CausalWanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    start_frame: int = 0,
    **kwargs,
):
    return dit(
        x=latents, 
        timestep=timestep, 
        context=context, 
        clip_feature=clip_feature, 
        y=y, 
        pose_latents=pose_latents,
        face_pixel_values=face_pixel_values,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        start_frame=start_frame,
        **kwargs,
    )

