#!/usr/bin/env python
"""Streaming inference for the stage2 distilled causal generator.

Block-causal autoregressive rollout: three denoising steps per block of three
latent frames, a KV cache carried across blocks, and each block decoded and
appended to the output as soon as it exists -- so frames come out while later
blocks are still being generated, rather than after the whole clip.

Loading is three-part, and it has to be. A stage2 checkpoint holds only the 800
trainable LoRA tensors; `pose_patch_embedding` -- the sole path by which the
render reaches the model -- is frozen in stage2 and therefore absent from it.
Loading a stage2 checkpoint alone leaves that weight at the zero init from
wan_wrapper, `x = x + pose_patch_embedding(pose)` becomes the identity, and the
model silently ignores its only control signal. So:

    1. base Wan2.1-I2V-14B-720P          (diffsynth/wan_wrapper.py, WAN_MODEL_DIR)
    2. stage1 checkpoint                  -> LoRA + pose_patch_embedding
    3. stage2 generator LoRA              -> overlays the 800 distilled tensors

Usage (single GPU, on a compute node -- the login node's 8 GB vmem ulimit kills
the model load and decord):

  srun -A nvr_elm_llm -p batch_short -N1 --gpus-per-node 1 -t 0:30:00 \
    python stream_infer.py \
      --stage1-ckpt  .../runs/stage1_tf_aligned/step-10000.safetensors \
      --stage2-ckpt  .../runs/stage2_render_dmd/ckpt/step_250_generator_lora.pt \
      --render       path/to/render.mp4 \
      --ref          path/to/first_real_frame.png \
      --caption      "..." \
      --out          out.mp4
"""
import argparse, os, time
import numpy as np
import torch
from PIL import Image

from diffsynth.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from diffsynth.utils.data import save_video, VideoData
from trainer.lora_helper import add_lora_to_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", required=True, help="provides pose_patch_embedding")
    p.add_argument("--stage2-ckpt", default=None, help="distilled generator LoRA; omit to run stage1 alone")
    p.add_argument("--render", required=True, help="low-poly control video")
    p.add_argument("--ref", required=True, help="one real frame, the appearance anchor")
    p.add_argument("--caption", required=True)
    p.add_argument("--out", default="stream_out.mp4")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--num-frames", type=int, default=49, help="4k+1; stage1 trained at 49")
    p.add_argument("--frames-per-block", type=int, default=3, help="latent frames per block")
    p.add_argument("--denoising-steps", type=float, nargs="+", default=[1000.0, 750.0, 500.0],
                   help="three, per the paper")
    p.add_argument("--lora-rank", type=int, default=128)
    p.add_argument("--lora-targets", default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--context-noise", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=24)
    return p.parse_args()


def main():
    a = parse_args()
    dev, dtype = "cuda", torch.bfloat16
    torch.set_grad_enabled(False)

    lat_f = (a.num_frames - 1) // 4 + 1
    n_blocks, rem = divmod(lat_f - 1, a.frames_per_block)
    if rem:
        raise ValueError(f"num_frames {a.num_frames} -> {lat_f} latent frames; "
                         f"content {lat_f-1} is not a multiple of {a.frames_per_block}")
    print(f"[setup] {a.num_frames} frames -> {lat_f} latent ({n_blocks} blocks of "
          f"{a.frames_per_block}) x {len(a.denoising_steps)} steps")

    # ---- 1+2+3: base, then stage1 (pose_patch_embedding), then stage2 --------
    gen = WanDiffusionWrapper(is_causal=True, local_attn_size=-1,
                              num_frame_per_block=a.frames_per_block)
    gen.model = add_lora_to_model(gen.model, lora_rank=a.lora_rank,
                                 lora_target_modules=a.lora_targets,
                                 pretrained_lora_path=a.stage1_ckpt)
    if a.stage2_ckpt:
        sd = torch.load(a.stage2_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = gen.model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"stage2 ckpt has {len(unexpected)} unexpected keys, "
                               f"e.g. {list(unexpected)[:4]}")
        print(f"[load] stage2 LoRA: {len(sd)} tensors overlaid")
        # The contract, checked rather than assumed.
        w = dict(gen.model.named_parameters()).get("pose_patch_embedding.weight")
        if w is not None and float(w.abs().max()) == 0.0:
            raise RuntimeError("pose_patch_embedding is all zeros: the render "
                               "conditioning would be a no-op. Did the stage1 "
                               "checkpoint load?")
    gen = gen.to(dev, dtype).eval()

    text_enc = WanTextEncoder().to(dev, dtype).eval()
    vae = WanVAEWrapper().to(dev, dtype).eval()

    # ---- conditioning -------------------------------------------------------
    ref = Image.open(a.ref).convert("RGB").resize((a.width, a.height))
    ref_t = (torch.from_numpy(np.array(ref)).float() / 127.5 - 1.0).permute(2, 0, 1)[None]
    render = VideoData(a.render).raw_data()[:a.num_frames]
    if len(render) < a.num_frames:
        raise ValueError(f"render has {len(render)} frames, need {a.num_frames}")
    render = [f.resize((a.width, a.height)) for f in render]
    render_t = torch.stack([(torch.from_numpy(np.array(f)).float() / 127.5 - 1.0).permute(2, 0, 1)
                            for f in render])[None].permute(0, 2, 1, 3, 4)

    ctx = text_enc([a.caption])["context"].to(dev, dtype)
    pose_lat = vae.encode_to_latent(render_t.to(dev, dtype))
    pose_lat = pose_lat[:, 1:].permute(0, 2, 1, 3, 4)          # drop the ref slot, as training does
    ref_lat = vae.encode_to_latent(ref_t[:, :, None].to(dev, dtype))

    cond = {"context": ctx, "pose_latents": pose_lat, "face_pixel_values": None,
            "y": torch.zeros_like(ref_lat).repeat(1, 1, lat_f, 1, 1),
            "clip_feature": None, "sink_flag": False}

    # ---- KV caches ----------------------------------------------------------
    nb = len(gen.model.blocks)
    fsl = (a.height // 8 // 2) * (a.width // 8 // 2)
    kv = [{"k": torch.zeros([1, lat_f * fsl, 12, 128], dtype=dtype, device=dev),
           "v": torch.zeros([1, lat_f * fsl, 12, 128], dtype=dtype, device=dev),
           "global_end_index": torch.tensor([0], dtype=torch.long, device=dev),
           "local_end_index": torch.tensor([0], dtype=torch.long, device=dev)} for _ in range(nb)]
    xattn = [{"k": torch.zeros([1, 0, 12, 128], dtype=dtype, device=dev),
              "v": torch.zeros([1, 0, 12, 128], dtype=dtype, device=dev),
              "is_init": False} for _ in range(nb)]

    # ---- block-by-block rollout, emitting as it goes ------------------------
    sched = gen.scheduler
    steps = [torch.tensor(float(s)) for s in a.denoising_steps]
    out_frames, start, t0 = [], 0, time.time()
    for b in range(n_blocks):
        n = a.frames_per_block
        x = torch.randn([1, n, 16, a.height // 8, a.width // 8], device=dev, dtype=dtype)
        for i, ts in enumerate(steps):
            tvec = torch.ones([1, n], device=dev, dtype=torch.int64) * int(ts)
            _, x0 = gen(noisy_image_or_video=x, conditional_dict=cond, timestep=tvec,
                        kv_cache=kv, crossattn_cache=xattn, current_start=start * fsl)
            if i + 1 < len(steps):
                nt = int(steps[i + 1])
                x = sched.add_noise(
                    x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                    nt * torch.ones([n], device=dev, dtype=torch.long)
                ).unflatten(0, x0.shape[:2])

        # Write the finished block into the cache at (near-)zero noise, so the
        # next block attends to what was actually produced -- this is the
        # self-forcing condition the student was distilled under.
        ct = torch.ones_like(tvec) * int(a.context_noise)
        cached = sched.add_noise(
            x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
            int(a.context_noise) * torch.ones([n], device=dev, dtype=torch.long)
        ).unflatten(0, x0.shape[:2]) if a.context_noise > 0 else x0
        gen(noisy_image_or_video=cached, conditional_dict=cond, timestep=ct,
            kv_cache=kv, crossattn_cache=xattn, current_start=start * fsl)

        px = vae.decode_to_pixel(x0.permute(0, 2, 1, 3, 4))
        px = ((px.float().clamp(-1, 1) + 1) * 127.5).byte()[0].permute(1, 2, 3, 0).cpu().numpy()
        out_frames.extend(list(px))
        start += n
        el = time.time() - t0
        print(f"[block {b+1}/{n_blocks}] {len(out_frames)} frames, "
              f"{el:.1f}s, {len(out_frames)/el:.2f} fps", flush=True)

    save_video(out_frames, a.out, fps=a.fps, quality=5)
    print(f"[done] {len(out_frames)} frames -> {a.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
