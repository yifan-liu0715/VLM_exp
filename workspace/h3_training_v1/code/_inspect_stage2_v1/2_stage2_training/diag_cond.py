#!/usr/bin/env python
"""Bisect: is the noise coming from my hand-built batch, or from the rollout?

The rollout is now the training pipeline itself, so if a batch produced by the
real RenderVideoDataset decodes to a sane image while my hand-built batch does
not, the bug is entirely in how I assemble the batch -- and the shape dump below
says exactly where.

Runs both through the identical build_conditioning + inference_with_trajectory.
"""
import os, sys, json
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from diffsynth.wan_wrapper import (WanDiffusionWrapper, WanTextEncoder,
                                   WanVAEWrapper, WanImageEncoderWrapper)
from diffsynth.utils.data import save_video, VideoData
from trainer.lora_helper import add_lora_to_model
from datasets.data_v2v_pose import prepare_multimodal_batch
from trainer.self_forcing_training import SelfForcingTrainingPipeline
from long_infer import build_conditioning, to_tensor, load_generator

RUN = "/home/yuchaog/workplace/code/yiren/project/WAN14B"
CFG = "configs/render_dmd.yaml"
H, W = 480, 848


def dump(tag, cond):
    print(f"\n--- cond dict: {tag} ---")
    for k, v in cond.items():
        if torch.is_tensor(v):
            print(f"  {k:<20} {tuple(v.shape)}  {str(v.dtype):<16} "
                  f"min {v.float().min():8.3f} max {v.float().max():8.3f} "
                  f"mean {v.float().mean():8.3f}")
        else:
            print(f"  {k:<20} {type(v).__name__}  {v if not hasattr(v,'__len__') else ''}")
    sys.stdout.flush()


def run_one(tag, cond, pipeline, vae, a, out_png):
    lat_f = cond["pose_latents"].shape[2]
    n_lat = (lat_f // 3) * 3
    print(f"[{tag}] pose_latents T={lat_f} -> rolling {n_lat} latent frames", flush=True)
    noise = torch.randn([1, n_lat, 16, H // 8, W // 8], device="cuda", dtype=torch.bfloat16)
    out = pipeline.inference_with_trajectory(noise=noise, slice_last_frames=12, **cond)
    lat = out[0] if isinstance(out, tuple) else out
    print(f"[{tag}] latent out {tuple(lat.shape)} "
          f"min {lat.float().min():.3f} max {lat.float().max():.3f} "
          f"std {lat.float().std():.3f}", flush=True)
    px = vae.decode_to_pixel(lat[:, :7])
    img = ((px.float().clamp(-1, 1) + 1) * 127.5).byte()[0].permute(0, 2, 3, 1).cpu().numpy()
    Image.fromarray(img[4]).save(out_png)
    print(f"[{tag}] frame4 mean {img[4].mean():.1f} std {img[4].std():.1f} -> {out_png}",
          flush=True)


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
    torch.set_grad_enabled(False)

    cfg = yaml.safe_load(open(CFG))
    mk = dict(cfg.get("model_kwargs", {}))

    class A: pass
    a = A()
    a.lora_rank, a.lora_targets, a.frames_per_block = 128, "q,k,v,o,ffn.0,ffn.2", 3
    a.stage1_ckpt = f"{RUN}/runs/stage1_tf_aligned/step-10000.safetensors"
    a.stage2_ckpt = sorted(
        [f"{RUN}/runs/stage2_render_dmd/ckpt/{f}" for f in
         os.listdir(f"{RUN}/runs/stage2_render_dmd/ckpt") if f.endswith("_generator_lora.pt")],
        key=lambda p: int(p.split("step_")[1].split("_")[0]))[-1]
    print(f"[ckpt] {a.stage2_ckpt}", flush=True)

    gen = load_generator(a, "cuda", torch.bfloat16, mk)
    text_enc = WanTextEncoder().to("cuda", torch.bfloat16).eval()
    vae = WanVAEWrapper().to("cuda", torch.bfloat16).eval()
    img_enc = WanImageEncoderWrapper().to("cuda", torch.bfloat16).eval()

    pipeline = SelfForcingTrainingPipeline(
        denoising_step_list=[float(x) for x in cfg["denoising_step_list"]],
        scheduler=gen.get_scheduler(), generator=gen, num_frame_per_block=3,
        independent_first_frame=False, same_step_across_blocks=True,
        last_step_only=False, num_max_frames=12,
        rerun_kv_cache=cfg.get("rerun_kv_cache", True),
        context_noise=cfg.get("context_noise", 0),
        local_attn_size=mk.get("local_attn_size", -1),
        slice_last_frames=12, num_training_frames=12, vae=vae,
        target_hw_buckets=[(H, W)])
    print(f"[pipeline] fsl={pipeline.frame_seq_length} kv={pipeline.kv_cache_size}", flush=True)

    # ---- A: the real dataset, exactly as training feeds it -------------------
    from datasets.data_render import RenderVideoDataset
    vt = cfg["v2v_training_config"]
    ds_kw = dict(vt["dataset"])
    for k in ("enable_pose_keypoints", "pose_checkpoint", "detector_checkpoint"):
        ds_kw.pop(k, None)
    ds = RenderVideoDataset.make_datasets(vt["dataset_path"], type("C", (), vt)(),
                                          infer=False, **ds_kw)
    s = ds[0]
    print("\n--- RenderVideoDataset sample keys/shapes ---")
    for k, v in s.items():
        print(f"  {k:<20} {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}")
    batch_ds = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in s.items()}
    batch_ds.setdefault("tgt_frame_face", None)
    _, cd_ds, _, _ = prepare_multimodal_batch(batch_ds, "cuda", torch.bfloat16, neg_cond_prompt="")
    print("\n--- prepare_multimodal_batch (dataset) ---")
    for k, v in cd_ds.items():
        print(f"  {k:<20} {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}")
    cond_ds = build_conditioning(cd_ds, text_enc, img_enc, vae, "cuda", torch.bfloat16)
    dump("dataset", cond_ds)
    run_one("dataset", cond_ds, pipeline, vae, a, "/tmp/diag_dataset.png")

    # ---- B: my hand-built batch from the boomerang mp4 ------------------------
    man = [json.loads(l) for l in open(f"{RUN}/runs/infer10s/prep/manifest.jsonl")][0]
    render = [f.resize((W, H), Image.BICUBIC) for f in VideoData(man["render"]).raw_data()][:49]
    batch_mine = {
        "tgt_frame": torch.stack([to_tensor(f) for f in render])[None],
        "tgt_frame_pose": torch.stack([to_tensor(f) for f in render])[None],
        "tgt_frame_face": None,
        "ref_image": to_tensor(Image.open(man["ref"]).convert("RGB"))[None][None],
        "text": [man["caption"]],
    }
    print("\n--- my hand-built batch ---")
    for k, v in batch_mine.items():
        print(f"  {k:<20} {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}")
    _, cd_mine, _, _ = prepare_multimodal_batch(batch_mine, "cuda", torch.bfloat16, neg_cond_prompt="")
    cond_mine = build_conditioning(cd_mine, text_enc, img_enc, vae, "cuda", torch.bfloat16)
    dump("mine", cond_mine)
    run_one("mine", cond_mine, pipeline, vae, a, "/tmp/diag_mine.png")


if __name__ == "__main__":
    main()
