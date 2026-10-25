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

from ..models.wan_video_dit_causal_animate import CausalWanModel, sinusoidal_embedding_1d
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
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
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
        text_encoder_state_dict = load_state_dict(model_paths[1], torch_dtype=torch.bfloat16, device="cpu")
        text_encoder_model_instance = WanTextEncoder()
        text_encoder_model_instance.load_state_dict(text_encoder_state_dict, strict=True)
        pipe.text_encoder = text_encoder_model_instance.to(device=device, dtype=torch_dtype)
        
        # Image encoder
        clip_state_dict = load_state_dict(model_paths[3], torch_dtype=torch.bfloat16, device="cpu")
        fixed_state = {}
        for k, v in clip_state_dict.items():
            new_k = k if k.startswith("model.") else "model." + k
            fixed_state[new_k] = v
        filtered = {k: v for k, v in fixed_state.items() if k.startswith("model.visual.") or k == "model.log_scale"}
        clip_model_instance = WanImageEncoder()
        clip_model_instance.load_state_dict(filtered, strict=True)
        pipe.image_encoder = clip_model_instance.to(device=device, dtype=torch_dtype)
        
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
        state_dict = load_state_dict(file_path=model_paths[0], torch_dtype=torch.bfloat16, device="cpu")
        model = CausalWanModel(**kwargs)
        model.load_state_dict(state_dict, strict=True)
        pipe.dit = model.to(device=device, dtype=torch_dtype)
        
        # vae
        vae_state_dict = load_state_dict(model_paths[2], torch_dtype=torch.bfloat16, device="cpu")
        fixed_state = {}
        for k, v in vae_state_dict.items():
            new_k = k if k.startswith("model.") else "model." + k
            fixed_state[new_k] = v
        vae_model_instance = WanVideoVAE()
        vae_model_instance.load_state_dict(fixed_state, strict=True)        
        pipe.vae = vae_model_instance.to(device=device, dtype=torch_dtype)
        
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
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # VAP
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
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
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[list[Image.Image]] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "vap_video": vap_video, 
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
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            ref_latents = inputs_shared["latents"][:, :, 0:1]  # First frame latents
            content_latents = inputs_shared["latents"][:, :, 1:]  # Other frames latents
            content_latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], content_latents) # only denoise content latents
            inputs_shared["latents"] = torch.concat([ref_latents, content_latents], dim=2) # Combine first frame latents back
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
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
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "input_image", "height", "width"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, input_image, height, width):
        if input_video is None:
            if input_image is not None: # (1, F, C, H, W) (1, C, F, H, W) 
                if isinstance(input_image, Image.Image):
                    input_image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
                ref_latents = pipe.vae.encode(input_image.transpose(0,1).unsqueeze(0), device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise = torch.concat([ref_latents, noise], dim=2) # (1, C, F+1, H, W)
            return {"latents": noise}
        
        pipe.load_models_to_device(self.onload_model_names)
        
        if isinstance(input_video, list):
            input_video = pipe.preprocess_video(input_video)
        else:
            input_video = rearrange(input_video, "b t c h w -> b c t h w")
            
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

        if input_image is not None: # (1, F, C, H, W) (1, C, F, H, W) 
            input_image_latents = pipe.vae.encode(input_image.transpose(0,1).unsqueeze(0), device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([input_image_latents, input_latents], dim=2)    
        
        if pipe.scheduler.training:
            latents = torch.concat([input_latents[:, :, :1], noise], dim=2)
            return {"latents": latents, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}

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
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
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

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        if prompt is None:
            return {}
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
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        
        if isinstance(input_image, Image.Image):
            input_image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)

        image_input = input_image.transpose(0, 1)
        image_lat = pipe.vae.encode([image_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        padding_content = torch.zeros(3, num_frames, height, width).to(input_image.device)
        video_lat = pipe.vae.encode([padding_content.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]

        y = torch.concat([image_lat, video_lat], dim=-3)
        
        _,T,H,W = y.shape
        msk = torch.ones(4, T, H, W, device=pipe.device, dtype=pipe.torch_dtype)
        msk[:, 1:] = 0
        
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


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
        
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: CausalWanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: CausalWanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    pose_latents=None,
    face_pixel_values=None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    start_frame: int = 0,
    **kwargs,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    cond_t_mod = torch.zeros_like(t_mod, dtype=t_mod.dtype, device=t_mod.device)

    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
        
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
        
    # Camera control
    x = dit.patchify(x, control_camera_latents_input)
    
    ref = x[:, :, 0:1]
    x = x[:, :, 1:]
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = dit.after_patch_embedding(x, pose_latents, face_pixel_values)
    x = torch.cat([ref, x], dim=2)
    
    # Patchify
    f, h, w = x.shape[2:]

    def recompute_freqs(f, h, w):
        seq_end = f * h * w          # 所有帧的 token 数 (f 帧)
        ref_seq_len = h * w          # 参考帧的 token 数 (1 帧)
        content_seq_len = seq_end - ref_seq_len  # 内容帧的 token 数 (f-1 帧)
        freqs_ref = torch.cat([
                        dit.freqs[0][30:31].view(1, 1, 1, -1).expand(1, h, w, -1),  # temporal
                        dit.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),  # height
                        dit.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1)   # width
                    ], dim=-1).reshape(ref_seq_len, 1, -1).to(x.device)
        
        freqs_content = torch.cat([
                        dit.freqs[0][start_frame:start_frame + f - 1].view(f-1, 1, 1, -1).expand(f-1, h, w, -1),  # temporal
                        dit.freqs[1][:h].view(1, h, 1, -1).expand(f-1, h, w, -1),  # height
                        dit.freqs[2][:w].view(1, 1, w, -1).expand(f-1, h, w, -1)   # width
                    ], dim=-1).reshape(content_seq_len, 1, -1).to(x.device)
        freqs = torch.cat([freqs_ref, freqs_content], dim=0).to(x.device)
        return freqs
    freqs = recompute_freqs(f, h, w)
    
    num_frames = f
    frame_seqlen = h * w
    mask_key = (num_frames, h, w, dit.num_frame_per_block)
    if mask_key not in dit.block_mask_cache:
        dit.block_mask_cache[mask_key] = dit._prepare_blockwise_causal_attn_mask_i2v(
            device=x.device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=dit.num_frame_per_block,
            ref_seqlen=frame_seqlen,
            local_attn_size=-1
        )
    dit.block_mask = dit.block_mask_cache[mask_key]
    
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, cond_t_mod, freqs,
                        frame_seqlen=frame_seqlen,
                        block_mask=dit.block_mask,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, cond_t_mod, freqs,
                    frame_seqlen=frame_seqlen,
                    block_mask=dit.block_mask,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x, context, t_mod, cond_t_mod, freqs,
                    frame_seqlen=frame_seqlen,
                    block_mask=dit.block_mask,
                    )
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                ref = x[:, :frame_seqlen]
                content = x[:, frame_seqlen:]
                content = dit.after_transformer_block(block_id, content, motion_vec)
                x = torch.cat([ref, content], dim=1)
                
        if tea_cache is not None:
            tea_cache.store(x)
                
    x = x[:,frame_seqlen:]  # remove reference frame tokens
    x = dit.head(x, t)
    
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x

    x = dit.unpatchify(x, (f-1, h, w))
    return x
