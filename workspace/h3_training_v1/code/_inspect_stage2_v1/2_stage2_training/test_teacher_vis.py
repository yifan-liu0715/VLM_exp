"""
Teacher (WanModel bidirectional) 推理可视化测试。
数据流与 distillation.py 完全对齐，支持 LoRA checkpoint 加载。

用法：
    python test_teacher_vis.py \
        --dataset_path /path/to/merged.jsonl \
        --output_dir ./test_teacher_vis_out \
        --num_steps 10 \
        --num_samples 3 \
        [--lora_ckpt /path/to/lora.safetensors] \
        [--lora_rank 128]
"""

import sys, os, json, copy, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from tqdm import tqdm
from einops import rearrange
from PIL import Image

from datasets.data_v2v import AnimateVideoDataset, prepare_multimodal_batch
from diffsynth.wan_wrapper import (
    WanVAEWrapper, WanTextEncoder, WanImageEncoderWrapper,
    init_dit_model, get_dit_state_dict
)
from diffsynth.scheduler import FlowMatchScheduler
from trainer.lora_helper import add_lora_to_model

try:
    from diffsynth.utils.data import save_video
except ImportError:
    import imageio, numpy as np
    def save_video(frames, path, fps=16, quality=5):
        writer = imageio.get_writer(path, fps=fps, quality=quality)
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()


# ─── 默认路径 ──────────────────────────────────────────────────────────────────
_DEFAULT_DATASET = "/primus_chat_aigc_sh_oss/users/zhangyuxuan/T08temp/liveanimate_v4/merged.jsonl"


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",       type=str, default=_DEFAULT_DATASET)
    p.add_argument("--lora_ckpt",          type=str, default="/primus_xpfs_workspace_T08/zhangyuxuan/projects2/LiveAnimate_stage1_0411/output_fsdp_lora_multinode_bid_tsf/step-2000.safetensors",
                   help="teacher LoRA checkpoint 路径")
    p.add_argument("--lora_rank",          type=int, default=128)
    p.add_argument("--lora_target_modules",type=str, default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--output_dir",  type=str, default="./test_teacher_vis_out")
    p.add_argument("--num_steps",   type=int, default=10)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--fps",         type=int, default=16)
    p.add_argument("--target_len",  type=int, default=77)
    return p


# ─── 模型加载 ──────────────────────────────────────────────────────────────────

def load_teacher(lora_ckpt, lora_rank, lora_target_modules, device):
    print("[teacher] 加载 bidirectional WanModel...")
    # 复用全局缓存，避免重复读盘
    state_dict = get_dit_state_dict()
    teacher = init_dit_model(is_causal=False, state_dict=state_dict)

    if lora_ckpt is not None:
        print(f"[teacher] 注入 LoRA 结构并加载权重: {lora_ckpt}")
        teacher = add_lora_to_model(
            teacher,
            lora_rank=lora_rank,
            lora_target_modules=lora_target_modules,
            pretrained_lora_path=None,  # 先只注入结构
        )
        # LoRA ckpt 里 key 带 "model." 前缀，teacher 是裸 WanModel 不带，手动去掉
        from diffsynth.core import load_state_dict as _load_sd
        lora_state = _load_sd(lora_ckpt, torch_dtype=torch.bfloat16)
        # 打印前几个 key 帮助诊断前缀问题
        ckpt_keys = list(lora_state.keys())
        print(f"[debug] LoRA ckpt 前3个key: {ckpt_keys[:3]}")
        model_keys = [k for k in teacher.state_dict().keys() if 'lora' in k]
        print(f"[debug] 模型 LoRA 前3个key: {model_keys[:3]}")
        lora_state_fixed = {}
        for k, v in lora_state.items():
            new_k = k[len("model."):] if k.startswith("model.") else k
            lora_state_fixed[new_k] = v
        missing, unexpected = teacher.load_state_dict(lora_state_fixed, strict=False)
        if unexpected:
            print(f"[teacher] LoRA unexpected keys: {len(unexpected)}, 前3个: {unexpected[:3]}")
        if missing:
            print(f"[teacher] LoRA missing keys: {len(missing)}, 前3个: {missing[:3]}")
        print("[teacher] LoRA 加载完成")
    else:
        print("[teacher] 无 LoRA checkpoint，使用 base model 权重")

    teacher.requires_grad_(False)
    teacher.eval()
    teacher.use_gradient_checkpointing = True
    teacher.to(device=device, dtype=torch.bfloat16)
    # peft inject 后部分层可能是 float32，强制统一
    for p in teacher.parameters():
        p.data = p.data.to(torch.bfloat16)
    print("[teacher] 加载完成")
    return teacher


# ─── 推理（数据流与 distillation.py 完全对齐）────────────────────────────────

@torch.no_grad()
def run_teacher(teacher, vae, text_encoder, image_encoder,
                cond_dict, scheduler, num_steps, device):
    dtype = torch.bfloat16

    # ── 与 distillation.py 完全一致的条件处理 ─────────────────────────────────
    # text
    context = text_encoder(text_prompts=cond_dict["prompt"], context=None).to(device)

    # pose: encode 后 [:,1:] 去掉第一个 latent 帧
    pose = vae.encode_to_latent(cond_dict["pose"])   # B T_lat C H W
    pose = pose[:, 1:]
    psi_pose = rearrange(pose, 'b t c h w -> b c t h w').to(device, dtype=dtype)

    # face: pixel space，去掉第一帧
    cond_dict["face"] = cond_dict["face"][:, :, 1:]

    # clip
    clip_feature = image_encoder(images=cond_dict["ref_image"]).to(device, dtype=dtype)

    # ref latent: encode_to_latent 返回 B T C H W，转成 B C T H W
    image_lat = vae.encode_to_latent(
        cond_dict["ref_image"].unsqueeze(2)  # B C 1 H W
    ).to(device, dtype=dtype)  # B 1 C H W
    image_lat = rearrange(image_lat, 'b t c h w -> b c t h w')  # B C 1 H W

    # y 构造：与训练 pipeline WanVideoUnit_ImageEmbedderVAE 完全一致
    # ref 位置和 content 位置都是全零 VAE latent，mask 全零
    _, _, _, H_lat, W_lat = image_lat.shape  # B C 1 H W
    B = image_lat.shape[0]
    T_content_lat = pose.shape[1]
    padding_ref = torch.zeros(B, 3, 1, H_lat * 8, W_lat * 8, device=device, dtype=dtype)
    padding_content = torch.zeros(B, 3, T_content_lat * 4 + 1, H_lat * 8, W_lat * 8, device=device, dtype=dtype)
    ref_lat_y = vae.encode_to_latent(padding_ref)          # B 1 C H W
    content_lat_y = vae.encode_to_latent(padding_content)  # B T_lat C H W
    content_lat_y = content_lat_y[:, 1:]                   # 去掉第一帧
    y_vae = torch.cat([ref_lat_y, content_lat_y], dim=1)   # B T C H W
    y_vae = rearrange(y_vae, 'b t c h w -> b c t h w').to(device, dtype=dtype)
    _, _, T_y, _, _ = y_vae.shape
    msk = torch.zeros(B, 4, T_y, H_lat, W_lat, device=device, dtype=dtype)
    y = torch.cat([msk, y_vae], dim=1)  # B (4+16) T H W

    # ── 初始化噪声 ────────────────────────────────────────────────────────────
    noisy_latents = torch.randn(
        B, 16, T_content_lat, H_lat, W_lat,
        device=device, dtype=dtype
    )

    # ── 多步去噪 ──────────────────────────────────────────────────────────────
    infer_scheduler = copy.deepcopy(scheduler)
    infer_scheduler.set_timesteps(num_steps)
    # 把 scheduler 的 sigmas/timesteps 移到 GPU，避免 step 时设备不一致
    infer_scheduler.sigmas    = infer_scheduler.sigmas.to(device)
    infer_scheduler.timesteps = infer_scheduler.timesteps.to(device)

    teacher_kwargs = dict(
        context=context,
        clip_feature=clip_feature,
        y=y,
        pose_latents=psi_pose,
        face_pixel_values=cond_dict["face"].to(device, dtype=dtype),
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    )

    noisy_latents = torch.cat([image_lat, noisy_latents], dim=2).to(device, dtype=dtype)  # B C T H W
    print(f"[debug] noisy_latents: {noisy_latents.shape}")
    print(f"[debug] psi_pose:      {psi_pose.shape}")
    print(f"[debug] face:          {cond_dict['face'].shape}")
    print(f"[debug] y:             {y.shape}")
    print(f"[debug] image_lat min/max: {image_lat.min():.3f} / {image_lat.max():.3f}")
    for progress_id, ts in enumerate(tqdm(infer_scheduler.timesteps, desc="teacher denoise", leave=False)):
        ts_t     = ts.to(device=device, dtype=dtype)
        timestep = ts_t.unsqueeze(0).expand(B)
        flow_pred = teacher(x=noisy_latents, timestep=timestep, **teacher_kwargs)
        if progress_id == 0:
            print(f"[debug] flow_pred:     {flow_pred.shape}")
        # flow_pred 是 content only，只对 content 部分做 step，ref 帧保持不变
        content_denoised = infer_scheduler.step(
            flow_pred, infer_scheduler.timesteps[progress_id].unsqueeze(0), noisy_latents[:, :, 1:]
        )
        noisy_latents = torch.cat([noisy_latents[:, :, :1], content_denoised], dim=2).to(device, dtype=dtype)  # B C T H W

    # ── decode（去掉 ref 帧）─────────────────────────────────────────────────
    # noisy_latents 是 B C T H W，decode_to_pixel 期望 B T C H W
    content_latents = noisy_latents[:, :, 1:]                          # B C T H W
    content_latents = rearrange(content_latents, 'b c t h w -> b t c h w')  # B T C H W
    video_tensor = vae.decode_to_pixel(content_latents)  # B T C H W
    video = ((video_tensor[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    frames = [Image.fromarray(video[i].permute(1, 2, 0).cpu().numpy()) for i in range(video.shape[0])]
    return frames


def tensor_to_frames(t):
    t = t.float().cpu()
    t = (t * 0.5 + 0.5).clamp(0, 1)
    t = (t * 255).byte()
    return [Image.fromarray(t[i].permute(1, 2, 0).numpy()) for i in range(t.shape[0])]


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    print("加载 VAE / TextEncoder / ImageEncoder ...")
    vae           = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)
    text_encoder  = WanTextEncoder().to(device=device, dtype=torch.bfloat16)
    image_encoder = WanImageEncoderWrapper().to(device=device, dtype=torch.bfloat16)

    teacher = load_teacher(
        args.lora_ckpt, args.lora_rank, args.lora_target_modules, device
    )

    scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)

    print(f"加载 dataset: {args.dataset_path}")
    dataset = AnimateVideoDataset(
        data_meta_path=args.dataset_path,
        max_num_frames=170,
        dataset_max_token=160000,
        max_token=160000,
        max_wh=720,
        enable_motion_frames=False,
        target_hw=[(512, 512), (720, 400), (400, 720)],
        target_len=args.target_len,
        motion_len=12,
    )
    print(f"Dataset 长度: {len(dataset)}")

    saved = 0
    idx   = 0
    while saved < args.num_samples and idx < len(dataset):
        sample = dataset[idx]
        idx += 1
        if sample is None:
            continue

        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
                 for k, v in sample.items()}

        print(f"\n[{saved+1}/{args.num_samples}] idx={idx-1}  video={sample.get('video_path','?')}")

        try:
            _, cond_dict, _, _ = prepare_multimodal_batch(
                batch, device, torch.bfloat16, neg_cond_prompt=""
            )

            frames = run_teacher(
                teacher, vae, text_encoder, image_encoder,
                cond_dict, scheduler,
                num_steps=args.num_steps, device=device
            )

            save_video(frames,
                       os.path.join(args.output_dir, f"teacher_{saved:02d}.mp4"),
                       fps=args.fps, quality=5)
            save_video(tensor_to_frames(sample["tgt_frame_pose"]),
                       os.path.join(args.output_dir, f"teacher_{saved:02d}_pose.mp4"),
                       fps=args.fps, quality=5)
            save_video(tensor_to_frames(sample["tgt_frame_face"]),
                       os.path.join(args.output_dir, f"teacher_{saved:02d}_face.mp4"),
                       fps=args.fps, quality=5)
            print(f"  已保存 ({len(frames)} 帧)")
            saved += 1

        except Exception as e:
            import traceback
            print(f"  [WARN] 失败: {e}")
            traceback.print_exc()

    print(f"\n完成，共保存 {saved} 个视频到 {args.output_dir}/")


if __name__ == "__main__":
    main()
