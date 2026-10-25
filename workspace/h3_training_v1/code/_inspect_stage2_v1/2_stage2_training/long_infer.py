#!/usr/bin/env python
"""Long-video block-causal inference for the stage2 distilled generator.

Rolls out arbitrarily many blocks of 3 latent frames, three denoising steps
each, carrying a KV cache across blocks. Written against this tree's own
training rollout (`trainer/self_forcing_training.py::inference_with_trajectory`)
so the inference-time conditions match what the student was distilled under --
the previous ad-hoc script did not, and would have failed on the very first
forward.

Three things the earlier version got wrong, all of them silent-ish:

  * `heads=40` for the 14B model, not 12 (`_initialize_kv_cache`).
  * The cache key is `k_raw`, and the dict also needs `cond_k`/`cond_v`/
    `cond_end` (the reference-frame tokens every block attends) and
    `sink_start`/`sink_end` (PR-Sink read pointers). A dict without them
    KeyErrors inside SelfAttention.
  * The reference frame reaches the model only through the cond prefill that
    training runs when `sink_flag=True` (distillation.py:549). Skip it and
    `cond_end` stays 0, the ref tokens are never attended, and the model has no
    appearance anchor at all.

Memory is why this is not just "run more blocks". A full cache over 61 latent
frames at 848x480 is 61 x 1590 tokens x 40 heads x 128 dim x 2 B x 2 tensors x
40 layers = ~79 GB, before the 28 GB of model. So the cache is allocated for a
bounded window and the writes wrap: `current_start % k_raw.shape[1]` in
SelfAttention turns it into a ring buffer, i.e. a sliding window over recent
history, while the reference tokens stay pinned in cond_k/cond_v and remain
visible from every block. That is what makes 10 s fit on one GPU.

--kv-frames defaults to 12, matching `num_training_frames: 12`: the model has
never attended to a longer history, so a larger window is extrapolation. It is
exposed because it is worth trying, not because it is known to help.

Single GPU is enough and schedules far faster than 8:

  srun -A nvr_elm_llm -p batch_short -N1 --gpus-per-node 1 -t 1:00:00 \
    python long_infer.py --manifest <prep>/manifest.jsonl --out-dir <out> \
      --stage1-ckpt .../stage1_tf_aligned/step-10000.safetensors \
      --stage2-ckpt .../stage2_render_dmd/ckpt/step_2200_generator_lora.pt
"""
import argparse, json, os, time
import yaml
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from einops import rearrange

from diffsynth.wan_wrapper import (WanDiffusionWrapper, WanTextEncoder,
                                   WanVAEWrapper, WanImageEncoderWrapper)
from diffsynth.utils.data import save_video, VideoData
from trainer.lora_helper import add_lora_to_model
from datasets.data_v2v_pose import prepare_multimodal_batch
from trainer.self_forcing_training import SelfForcingTrainingPipeline

HEADS, DIM_HEAD = 40, 128          # 14B; see _initialize_kv_cache
COND_CACHE_TOKENS = 3000           # same allocation training uses


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="jsonl from make_boomerang.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stage1-ckpt", required=True, help="LoRA + pose_patch_embedding")
    p.add_argument("--stage2-ckpt", default=None, help="distilled generator LoRA")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--frames-per-block", type=int, default=3)
    p.add_argument("--denoising-steps", type=float, nargs="+", default=[1000.0, 750.0, 500.0])
    p.add_argument("--context-noise", type=int, default=0)      # config: context_noise: 0
    p.add_argument("--kv-frames", type=int, default=12,
                   help="KV cache capacity in latent frames; larger = fewer wraps")
    p.add_argument("--local-attn-size", type=int, default=-1,
                   help="attention span in latent frames. -1 keeps the training "
                        "config's full-cache read; a positive value turns on the "
                        "sliding window (12 == the span training actually had)")
    p.add_argument("--config", default="configs/render_dmd.yaml",
                   help="training config; model_kwargs and denoising_step_list "
                        "are read from it so inference cannot drift from training")
    p.add_argument("--lora-rank", type=int, default=128)
    p.add_argument("--lora-targets", default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--decode-chunk", type=int, default=13,
                   help="latent frames per VAE decode call; 0 = decode all at once")
    p.add_argument("--limit", type=int, default=0, help="only the first N clips")
    return p.parse_args()


def to_tensor(img):
    return (torch.from_numpy(np.array(img)).float() / 127.5 - 1.0).permute(2, 0, 1)


def load_generator(a, device, dtype, model_kwargs):
    # **model_kwargs from the training config, exactly as trainer/base.py:37 does
    # it. Hardcoding the signature defaults instead is what turned the output
    # into noise: WanDiffusionWrapper defaults timestep_shift=3.0 while this
    # config sets 1.0, and that shift defines the whole FlowMatchScheduler sigma
    # curve -- so add_noise() between the 1000/750/500 steps injected the wrong
    # amount at every intermediate step. The model was fine; the schedule was not.
    # local_attn_size is the one model_kwarg worth overriding at inference. With
    # -1 (the training value) SelfAttention reads the cache as
    # `k_raw[0:current_start]` (wan_video_dit_causal_animate.py:827-836) while the
    # write pointer is wrapped modulo the cache size -- so the moment
    # current_start wraps to 0 the model attends to NO history at all. That is a
    # hard reset every `kv_frames/3` blocks, not the sliding window the paper
    # describes (a three-slot Rolling Window holding t-2, t-1, t).
    # A positive value takes the `else` branch and computes a real sliding start,
    # which is what we want; the wrap still costs one reset per cache cycle, so
    # keep --kv-frames as large as memory allows to make cycles long.
    mk = dict(model_kwargs)
    if a.local_attn_size > 0:
        mk["local_attn_size"] = a.local_attn_size
    gen = WanDiffusionWrapper(**mk, is_causal=True,
                              num_frame_per_block=a.frames_per_block)
    # The WRAPPER, not gen.model -- exactly as distillation.py:125 does it. The
    # adapter then lands one level out, so parameter names carry the `model.`
    # prefix that the stage1 checkpoint's keys were saved with. Passing gen.model
    # here creates slots named `blocks.0...` against a checkpoint holding
    # `model.blocks.0...` and matches 0/802 tensors.
    gen = add_lora_to_model(gen, lora_rank=a.lora_rank,
                            lora_target_modules=a.lora_targets,
                            pretrained_lora_path=a.stage1_ckpt)
    if a.stage2_ckpt:
        sd = torch.load(a.stage2_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = gen.model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"stage2 ckpt: {len(unexpected)} unexpected keys, "
                               f"e.g. {list(unexpected)[:4]}")
        print(f"[load] stage2 LoRA overlaid: {len(sd)} tensors", flush=True)

    # The render reaches the model only through this weight, and stage2 does not
    # carry it. If stage1 failed to load, generation would silently ignore its
    # only control signal and produce an unconditional video that still looks
    # plausible -- so assert rather than trust.
    # The stage2 checkpoint, unlike stage1, was saved from `.model`, so its keys
    # have no `model.` prefix and it loads one level in. The two checkpoints
    # genuinely disagree on prefix; that is not a typo.
    named = dict(gen.named_parameters())
    w = named.get("model.pose_patch_embedding.weight",
                  named.get("pose_patch_embedding.weight"))
    if w is None:
        raise RuntimeError("no pose_patch_embedding in the model")
    if float(w.abs().max()) == 0.0:
        raise RuntimeError("pose_patch_embedding is all zeros -- the stage1 "
                           "checkpoint did not load; render conditioning would "
                           "be a no-op.")
    return gen.to(device, dtype).eval()


@torch.no_grad()
def build_conditioning(cd, text_enc, img_enc, vae, device, dtype):
    """The conditional dict, mirroring trainer/distillation.py:498-551.

    Kept line-for-line with training on purpose. Two things are easy to get
    wrong by writing it from intuition instead of from that code:

      * `y` is NOT the reference image. It is the VAE encoding of ZEROS -- a
        zero first frame plus a zero content video -- with a 4-channel zero mask
        prepended, so 20 channels, not 16. Feeding the reference here is both
        wrong content and a shape error.
      * The reference image reaches the model by exactly two paths: `ref_lat`,
        consumed by the sink prefill into cond_k/cond_v, and `clip_feature` from
        the CLIP image encoder. Dropping either silently removes the appearance
        anchor while still producing a plausible-looking video.
    """
    context = text_enc(text_prompts=cd["prompt"], context=None)

    pose = vae.encode_to_latent(cd["pose"])                    # B t C H W
    # distillation.py:509 `pose = pose[:,1:]` -- the model predicts latents
    # 1..N, latent 0 being the reference frame it is given. Without this drop
    # the render is fed to the model one latent frame (4 pixel frames) ahead of
    # where training put it, and pose_latents no longer lines up with `y`,
    # which *is* sliced (y[:, :, 1:]) after the sink prefill.
    pose = pose[:, 1:]
    psi_pose = rearrange(pose, "b t c h w -> b c t h w")

    clip_rea = img_enc(images=cd["ref_image"])
    image_lat = vae.encode_to_latent(cd["ref_image"].unsqueeze(2))   # B 1 C H W

    padding_image = torch.zeros_like(cd["ref_image"].unsqueeze(2))
    padding_content = torch.zeros_like(cd["pose"])
    padding_image_lat = vae.encode_to_latent(padding_image.to(dtype=dtype, device=device))
    padding_content_lat = vae.encode_to_latent(padding_content.to(dtype=dtype, device=device))[:, 1:]
    y = torch.concat([padding_image_lat, padding_content_lat], dim=1)
    y = rearrange(y, "b t c h w -> b c t h w")
    _, _, T, H, W = y.shape
    msk = torch.zeros(1, 4, T, H, W, device=device, dtype=dtype)
    y = torch.concat([msk, y.to(dtype=dtype, device=device)], dim=1)  # 4 + 16 = 20 ch

    # encode_to_latent() ends in `.float()`, so pose/ref latents come back fp32
    # while the DiT is bf16. Training never casts them either -- FSDP's
    # MixedPrecision casts forward inputs for it. Single-GPU inference has no
    # FSDP, and torch.cat happily promotes the ref latent back up to fp32, so
    # both would reach a bf16 Conv3d as fp32 and raise "Input type (float) and
    # bias type (c10::BFloat16) should be the same". Cast here instead.
    return {
        "context": context.to(device, dtype),
        "pose_latents": psi_pose.to(device, dtype),
        "face_pixel_values": None,
        "clip_feature": clip_rea.to(device, dtype),
        "y": y.to(device, dtype),
        "sink_flag": True,
        "ref_lat": image_lat.to(device, dtype),
    }


@torch.no_grad()
def rollout(pipeline, vae, cond, a, lat_f, device, dtype):
    """Delegate to the training pipeline's own rollout.

    The hand-written version of this produced washed-out noise from the very
    first block while the identical weights sampled correctly during training.
    Rather than keep guessing which of the many conventions was off -- and three
    guesses (timestep_shift, crossattn cache shape, decode layout) each turned
    out not to be it -- call `inference_with_trajectory`, which IS the code that
    writes the known-good `visualizations/` samples. Whatever it does with
    timesteps, exit flags, RoPE roll distance, cache wrapping and the ref-frame
    prefill is by definition what the student was distilled under.
    """
    noise = torch.randn([1, lat_f, 16, a.height // 8, a.width // 8],
                        device=device, dtype=dtype)
    out = pipeline.inference_with_trajectory(
        noise=noise, slice_last_frames=a.kv_frames, **cond)
    output = out[0] if isinstance(out, tuple) else out

    del pipeline.kv_cache1
    torch.cuda.empty_cache()

    # decode_to_pixel takes (B, T, C, H, W) and returns (B, T, C, H, W); the
    # training visualisation indexes it the same way (self_forcing_training.py:399).
    lat = output
    chunks, total = [], lat.shape[1]
    step = a.decode_chunk if a.decode_chunk > 0 else total
    for st in range(0, total, step):
        e = min(st + step, total)
        px = vae.decode_to_pixel(lat[:, max(0, st - 1):e])
        px = ((px.float().clamp(-1, 1) + 1) * 127.5).byte()[0].permute(0, 2, 3, 1).cpu().numpy()
        chunks.append(px if st == 0 else px[1:])
    return np.concatenate(chunks, axis=0)


def main():
    a = parse_args()
    a.denoising_steps_explicit = a.denoising_steps != [1000.0, 750.0, 500.0]
    device, dtype = "cuda", torch.bfloat16
    torch.set_grad_enabled(False)
    os.makedirs(a.out_dir, exist_ok=True)

    clips = [json.loads(l) for l in open(a.manifest)]
    if a.limit:
        clips = clips[:a.limit]


    cfg = yaml.safe_load(open(a.config))
    model_kwargs = dict(cfg.get("model_kwargs", {}))
    if not a.denoising_steps_explicit:
        a.denoising_steps = [float(x) for x in cfg["denoising_step_list"]]
    print(f"[config] {a.config}: model_kwargs={model_kwargs} "
          f"denoising_step_list={a.denoising_steps}", flush=True)
    gen = load_generator(a, device, dtype, model_kwargs)
    text_enc = WanTextEncoder().to(device, dtype).eval()
    vae = WanVAEWrapper().to(device, dtype).eval()
    img_enc = WanImageEncoderWrapper().to(device, dtype).eval()

    # inference_with_trajectory has two `dist.get_rank() == 0 and DEBUG` guards
    # that evaluate get_rank() first, so a single-process group must exist even
    # though there is nothing to communicate.
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        dist.init_process_group("nccl", rank=0, world_size=1)

    pipeline = SelfForcingTrainingPipeline(
        # float(), NOT int(). self_forcing_training.py:254 builds the timestep as
        # `torch.ones(..., dtype=torch.int64) * current_timestep`. A python float
        # promotes that to float32; a python int leaves it int64. And
        # sinusoidal_embedding_1d (wan_video_dit_causal_animate.py:677) ends in
        # `x.to(position.dtype)` -- with an int64 timestep the whole 256-dim
        # cos/sin embedding, all of it in [-1,1], truncates to zeros. The DiT then
        # gets no timestep signal at all and every block comes out as washed-out
        # noise. Training reads the list straight from the yaml, where the values
        # are 1000.0/750.0/500.0, so it never hit this. The t=0 sink prefill and
        # the t=0 cache rerun survive it (cos 0 = 1, sin 0 = 0 are exact), which
        # is why the reference frame still looked right.
        denoising_step_list=[float(x) for x in a.denoising_steps],
        scheduler=gen.get_scheduler(),
        generator=gen,
        num_frame_per_block=a.frames_per_block,
        independent_first_frame=False,
        same_step_across_blocks=True,
        # True, not False: with False, generate_and_sync_list picks a RANDOM exit
        # index per rollout and the denoising loop breaks there
        # (self_forcing_training.py:113-122), so a clip can run 1 step instead of
        # 3 -- observed in infer_6394457: one clip did index 0,1,2 while another
        # did index 0 only, for all 20 blocks. Training wants that randomness;
        # inference wants all three steps, every time, deterministically.
        last_step_only=True,
        num_max_frames=a.kv_frames,
        rerun_kv_cache=cfg.get("rerun_kv_cache", True),
        context_noise=cfg.get("context_noise", 0),
        local_attn_size=(a.local_attn_size if a.local_attn_size > 0
                         else model_kwargs.get("local_attn_size", -1)),
        slice_last_frames=a.kv_frames,
        num_training_frames=a.kv_frames,
        vae=vae,
        target_hw_buckets=[(a.height, a.width)],
    )
    print(f"[pipeline] frame_seq_length={pipeline.frame_seq_length} "
          f"kv_cache_size={pipeline.kv_cache_size}", flush=True)

    # T5 (10.6 GiB on disk, ~11.4 GiB resident) and the CLIP image encoder
    # (~2.5 GiB) are each used exactly once per clip, at the very start, yet they
    # sat resident for the whole rollout -- 13.9 of the 42.7 GiB of non-cache
    # memory measured when kv-frames=30 hit OOM. Encode everything up front, then
    # drop them: that budget buys ~11 more latent frames of KV cache.
    pre = []
    for c in clips:
        render = [f.resize((a.width, a.height), Image.BICUBIC)
                  for f in VideoData(c["render"]).raw_data()]
        n_px = len(render)
        if (n_px - 1) % 4:
            n_px = (n_px - 1) // 4 * 4 + 1
            render = render[:n_px]
        batch = {
            "tgt_frame": torch.stack([to_tensor(f) for f in render])[None],
            "tgt_frame_pose": torch.stack([to_tensor(f) for f in render])[None],
            "tgt_frame_face": None,
            "ref_image": to_tensor(Image.open(c["ref"]).convert("RGB"))[None][None],
            "text": [c["caption"]],
        }
        _, cd, _, _ = prepare_multimodal_batch(batch, device, dtype, neg_cond_prompt="")
        pre.append(build_conditioning(cd, text_enc, img_enc, vae, device, dtype))
        print(f"[pre] {c['stem']}  {n_px} frames  "
              f"pose_latents {tuple(pre[-1]['pose_latents'].shape)}", flush=True)

    del text_enc, img_enc
    torch.cuda.empty_cache()
    print(f"[mem] text+image encoders freed; "
          f"{torch.cuda.memory_allocated()/2**30:.1f} GiB still allocated", flush=True)

    for ci, c in enumerate(clips):
        t0 = time.time()
        cond = pre[ci]
        lat_f = cond["pose_latents"].shape[2]
        n_lat = (lat_f // a.frames_per_block) * a.frames_per_block
        print(f"\n[{ci+1}/{len(clips)}] {c['stem']}  {lat_f} latent -> rolling {n_lat}",
              flush=True)
        frames = rollout(pipeline, vae, cond, a, n_lat, device, dtype)
        op = os.path.join(a.out_dir, f"{c['stem']}_gen10s.mp4")
        save_video(list(frames), op, fps=a.fps, quality=6)
        print(f"  -> {op}  {len(frames)} frames in {time.time()-t0:.1f}s", flush=True)

    print("\n[done]")


if __name__ == "__main__":
    main()
