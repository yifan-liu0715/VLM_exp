import sys
import torch
import types
import math
sys.path.append("..")
from typing import List, Optional
from einops import rearrange
from diffsynth.models.wan_video_dit_causal_animate import CausalWanModel as causal_wan_model
from diffsynth.models.wan_video_dit_bidirection_animate import WanModel as wan_model
from diffsynth.models.wan_video_vae import VideoVAE_ as vae_model
from diffsynth.models.wan_video_text_encoder import WanTextEncoder as text_encoder
from diffsynth.models.wan_video_image_encoder import WanImageEncoder as image_encoder
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
from diffsynth.scheduler import SchedulerInterface, FlowMatchScheduler
from diffsynth.core.loader.file import load_state_dict

# Paths were hardcoded to the original author's cluster and to
# Wan2.2-Animate-14B (4 DiT shards). This project trains on Wan2.1-I2V-14B-720P
# (7 shards); override with WAN_MODEL_DIR if the base model ever moves.
import os as _os, glob as _glob
_MODEL_DIR = _os.environ.get("WAN_MODEL_DIR", "/home/yuchaog/workplace/models/Wan2.1-I2V-14B-720P")
_DIT_SHARDS = sorted(_glob.glob(_os.path.join(_MODEL_DIR, "diffusion_pytorch_model-*-of-*.safetensors")))
if not _DIT_SHARDS:
    raise FileNotFoundError(f"no DiT shards under {_MODEL_DIR}; set WAN_MODEL_DIR")
model_paths = [
    _DIT_SHARDS,
    _os.path.join(_MODEL_DIR, "models_t5_umt5-xxl-enc-bf16.pth"),
    _os.path.join(_MODEL_DIR, "Wan2.1_VAE.pth"),
    _os.path.join(_MODEL_DIR, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
]
tokenizer_path = _os.path.join(_MODEL_DIR, "google", "umt5-xxl")
default_dtype = torch.bfloat16

def init_dit_model(is_causal: bool = True, local_attn_size: int = -1, state_dict=None, use_memory: bool = True):
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
    if state_dict is None:
        state_dict = load_state_dict(file_path=model_paths[0], torch_dtype=torch.bfloat16, device="cpu")
    if is_causal:
        model = causal_wan_model(local_attn_size=local_attn_size, **kwargs).to(dtype=torch.bfloat16)
    else:
        model = wan_model(**kwargs).to(dtype=torch.bfloat16)
    # A plain Wan 14B I2V checkpoint carries none of the Animate-only modules, so
    # strict=True crashes on it. Let exactly those be missing and initialize them
    # here -- anything else missing means the checkpoint does not match kwargs.
    # This mirrors the same fix already made on the stage1 side.
    animate_only = ("pose_patch_embedding", "motion_encoder", "face_adapter", "face_encoder")
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    unexplained = [k for k in missing if not k.startswith(animate_only)]
    if unexplained or unexpected:
        raise RuntimeError(
            f"DiT checkpoint mismatch. missing={unexplained[:8]} unexpected={list(unexpected)[:8]}")
    if missing:
        init_from_scratch = sorted({k.split(".")[0] for k in missing})
        for name in init_from_scratch:
            getattr(model, name).to(dtype=torch.bfloat16)
        # The pose branch is additive, so zero-init leaves the pretrained model
        # untouched at step 0 and lets the render control come up gradually.
        if "pose_patch_embedding" in init_from_scratch:
            torch.nn.init.zeros_(model.pose_patch_embedding.weight)
            torch.nn.init.zeros_(model.pose_patch_embedding.bias)
        print(f"[wan_wrapper] initialized from scratch (not in checkpoint): {init_from_scratch}")
    return model

# 全局缓存，只从磁盘读一次
_cached_dit_state_dict = None

def get_dit_state_dict():
    global _cached_dit_state_dict
    if _cached_dit_state_dict is None:
        print("[wan_wrapper] Loading DIT state dict from disk (once)...")
        _cached_dit_state_dict = load_state_dict(file_path=model_paths[0], torch_dtype=torch.bfloat16, device="cpu")
    return _cached_dit_state_dict

def init_vae_model():
    vae_state_dict = load_state_dict(model_paths[2], torch_dtype=torch.bfloat16, device="cpu")
    vae_model_instance = vae_model().to(dtype=torch.bfloat16)
    vae_model_instance.load_state_dict(vae_state_dict, strict=True)
    return vae_model_instance

def init_text_encoder_model():
    text_encoder_state_dict = load_state_dict(model_paths[1], torch_dtype=torch.bfloat16, device="cpu")
    text_encoder_model_instance = text_encoder().to(dtype=torch.bfloat16)
    text_encoder_model_instance.load_state_dict(text_encoder_state_dict, strict=True)
    return text_encoder_model_instance

def init_clip_model():
    clip_state_dict = load_state_dict(model_paths[3], torch_dtype=torch.bfloat16, device="cpu")
    fixed_state = {}
    for k, v in clip_state_dict.items():
        new_k = k if k.startswith("model.") else "model." + k
        fixed_state[new_k] = v
    filtered = {k: v for k, v in fixed_state.items() if k.startswith("model.visual.") or k == "model.log_scale"}
    clip_model_instance = image_encoder().to(dtype=torch.bfloat16)
    clip_model_instance.load_state_dict(filtered, strict=True)
    return clip_model_instance

def init_tokenizer():
    tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean='whitespace')
    return tokenizer


class WanTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = init_text_encoder_model().eval().requires_grad_(False).to(dtype=default_dtype)
        self.tokenizer = init_tokenizer()
        
    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()
        
    def forward(self, text_prompts: List[str], context: Optional[torch.Tensor] = None) -> dict:
        if context is None:
            return self.forward_text(text_prompts)
        else:
            return self.forward_context(context)
    
    def forward_text(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0
        return context
    
    def forward_context(self, context: torch.Tensor) -> dict:
        return context
    
    
class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_vae_model().eval().requires_grad_(False).to(dtype=default_dtype)
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: (B, C, T, H, W)
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        # output: (B, T, C, H, W)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # latent: (B, T, C, H, W)
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        # output: (B, C, T, H, W)
        return output


class WanImageEncoderWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = init_clip_model().eval().requires_grad_(False).to(dtype=default_dtype)
        
    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images is already expected to be a list by encode_image
        if not isinstance(images, list):
            images = [images]
        return self.image_encoder.encode_image(images)


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            timestep_shift=3.0,
            is_causal=False,
            local_attn_size=-1,
            num_frame_per_block=3,
    ):
        super().__init__()

        self.model = init_dit_model(is_causal, local_attn_size, state_dict=get_dit_state_dict()).to(dtype=default_dtype)        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len =  32760 # [1, 21, 16, 60, 104]
        self.num_frame_per_block = num_frame_per_block
        self.local_attn_size = local_attn_size
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
            Convert flow matching's prediction to x0 prediction.
            flow_pred: the prediction with shape [B, C, H, W]
            xt: the input noisy data with shape [B, C, H, W]
            timestep: the timestep with shape [B]

            pred = noise - x0
            x_t = (1-sigma_t) * x0 + sigma_t * noise
            we have x0 = x_t - sigma_t * pred
            see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, 
        conditional_dict: dict,
        timestep: torch.Tensor, 
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        is_sink: bool = False
    ) -> torch.Tensor:
        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        if kv_cache is not None:
            flow_pred = self.model(
                x=noisy_image_or_video.permute(0, 2, 1, 3, 4), # b f c h w -> b c f h w
                timestep=input_timestep, 
                **conditional_dict,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            )
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                timestep=input_timestep, **conditional_dict
            )

        if is_sink:
            return None
        
        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        # teacher 模型内部已去掉 ref 帧，flow_pred T 可能小于 noisy_image_or_video T
        # 取 noisy_image_or_video 最后 flow_pred.shape[1] 帧对齐
        xt_aligned = noisy_image_or_video[:, -flow_pred.shape[1]:]
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=xt_aligned.flatten(0, 1),
            timestep=timestep[:, -flow_pred.shape[1]:].flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
        self.model.num_frame_per_block = self.num_frame_per_block