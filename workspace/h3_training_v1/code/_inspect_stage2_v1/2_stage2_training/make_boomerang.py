#!/usr/bin/env python
"""Build 10 s boomerang control clips from the 5 s test renders.

The corpus is 5 s per clip, but we want to test a 10 s rollout. Concatenating
two copies of the same clip would put a hard cut at the seam -- the model would
see motion teleport back to frame 0 and the KV cache would carry the wrong
history across it. Playing it forward then backward instead makes the seam
continuous in both position and (up to a sign flip) velocity, so the only thing
the model has to survive is a direction reversal, not a discontinuity.

Frame budget: the VAE needs 4k+1 frames, so 241 frames at 24 fps = 10.04 s.
Forward takes 121, the reverse takes the remaining 120 -- and the reverse skips
its own first frame (which would duplicate the forward's last) so the turnaround
is not a held pair.

Also writes the appearance reference. The render lags its source by
`render_lag_frames` (from Render/scripts/measure_offsets.py; see
Render/HANDOFF_dataset.md), and the dataset pairs a real frame at time t with
render frame round(t*24) + lag. So the boomerang starts at render[lag] and the
reference is real[0] -- starting at render[0] instead would need a real frame at
negative time, and the mismatched reference it got made the opening blocks much
worse than the later ones, which recover once the rollout builds its own history.
"""
import argparse, json, os
import numpy as np
from PIL import Image

from diffsynth.utils.data import VideoData, save_video


def boomerang(frames, total):
    """forward + reverse, seam-continuous, exactly `total` frames."""
    fwd_n = total // 2 + 1                       # 121 for total=241
    if len(frames) < fwd_n:
        idx = np.linspace(0, len(frames) - 1, fwd_n).round().astype(int)
        fwd = [frames[i] for i in idx]
    else:
        fwd = frames[:fwd_n]
    # Skip the last frame when reversing: fwd[-1] is already emitted, repeating
    # it would freeze one frame at the turnaround.
    rev = fwd[-2::-1][: total - fwd_n]
    out = fwd + rev
    assert len(out) == total, f"{len(out)} != {total}"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="/home/yuchaog/workplace/code/yiren/project/Render/dataset/metadata_test.jsonl")
    p.add_argument("--data-root", default="/home/yuchaog/workplace/code/yiren/project/Render/dataset")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-clips", type=int, default=5)
    p.add_argument("--frames", type=int, default=241, help="4k+1")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=480)
    a = p.parse_args()

    assert (a.frames - 1) % 4 == 0, "VAE needs 4k+1 frames"
    os.makedirs(a.out_dir, exist_ok=True)

    metas = [json.loads(l) for l in open(a.metadata)]
    picked, manifest = [], []
    seen_cat = set()
    for m in metas:                              # one per category, for variety
        c = m.get("category")
        if c in seen_cat:
            continue
        seen_cat.add(c)
        picked.append(m)
        if len(picked) == a.num_clips:
            break

    for i, m in enumerate(picked):
        uuid = m["uuid"]
        render_p = os.path.join(a.data_root, m["reference_video"])   # the geometric proxy
        real_p = os.path.join(a.data_root, m["video"])               # the real target
        lag = max(0, int(m.get("render_lag_frames", 15)))

        render = VideoData(render_p).raw_data()
        render = [f.resize((a.width, a.height), Image.BICUBIC) for f in render]

        # Drop the render's first `lag` frames. data_render.py:113-118 pairs a
        # real frame at absolute time t with render frame round(t*24) + offset,
        # and data_render.py:270 takes the reference as the window's FIRST REAL
        # frame. Starting the boomerang at render[0] therefore asks for the real
        # frame at t = -offset/24, which does not exist -- the reference shipped
        # with it was ~30 frames away from what render[0] depicts. Starting at
        # render[lag] makes t=0 the correct pairing, so the reference is real[0].
        render = render[min(lag, max(0, len(render) - 1)):]
        boom = boomerang(render, a.frames)

        real = VideoData(real_p).raw_data()
        ref = real[0].resize((a.width, a.height), Image.BICUBIC)

        stem = f"{i:02d}_{uuid[:8]}"
        rp = os.path.join(a.out_dir, f"{stem}_render10s.mp4")
        save_video([np.array(f) for f in boom], rp, fps=a.fps, quality=6)
        ref.save(os.path.join(a.out_dir, f"{stem}_ref.png"))

        manifest.append({"stem": stem, "uuid": uuid, "category": m.get("category"),
                         "render": rp, "ref": os.path.join(a.out_dir, f"{stem}_ref.png"),
                         "caption": m.get("caption", ""), "render_lag_frames": lag,
                         "src_render_frames": len(render), "render_start": lag})
        print(f"[{i+1}/{len(picked)}] {stem}  {m.get('category')}  "
              f"src={len(render)}f -> {a.frames}f  lag={lag}", flush=True)

    mp = os.path.join(a.out_dir, "manifest.jsonl")
    with open(mp, "w") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(manifest)} clips -> {mp}")


if __name__ == "__main__":
    main()
