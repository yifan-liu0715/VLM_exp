import os
import random

import decord
import numpy as np
import torch
from decord import VideoReader

from .data_v2v import AnimateVideoDataset


class RenderVideoDataset(AnimateVideoDataset):
    """
    Geometric reference video -> real video, mapped onto the Wan-Animate slots.

    Metadata is one JSON object per line with `video` and `reference_video`
    (paths relative to the jsonl's own directory) plus an inline `caption`
    string, as produced by Render/scripts/build_dataset.py.

        tgt_frame       <- video            the real clip, the target
        tgt_frame_pose  <- reference_video  the geometric proxy, the control
        ref_image       <- a frame of the real clip, see `ref_frame_mode`
        text            <- caption
        tgt_frame_face  not emitted; this corpus has no faces to crop and the
                         pipeline short-circuits the face branch when it is None

    Two things differ from AnimateVideoDataset and are handled here:

    - The two sides are not frame-aligned. The reference is always 124 frames
      at 24 fps (5.175 s) however long the real clip is and whatever its fps,
      so the sides are paired by absolute time rather than by frame index.
    - On top of that the render lags its source. Measured over all 9773 clips
      (Render/scripts/measure_offsets.py, motion-energy and spatial-correlation
      scores agreeing independently), 82% sit at +15/+16 frames and 10% sit at
      0, so a single global shift is wrong for a tenth of the corpus. The
      per-clip offsets live in clip_offsets.jsonl next to the metadata and are
      applied here; clips whose offset could not be measured confidently fall
      back to the corpus median, which is safe because an offset that cannot be
      measured is one where 15 frames of content barely differ.
    - The reference is a full-frame stretch of the real clip to a fixed
      1344x768, not a rescale of it. Decoding it straight to the real clip's
      scaled size undoes that stretch and lands both on the same pixel grid.
    """

    def __init__(self, data_meta_path, ref_frame_mode="window_first", offsets_path=None, **kwargs):
        kwargs.setdefault("enable_face_augment", False)
        super().__init__(data_meta_path=data_meta_path, **kwargs)
        # AnimateVideoDataset joins relative paths onto the jsonl path itself;
        # here they are relative to the directory holding the jsonl.
        self.basedir_name = os.path.dirname(os.path.abspath(data_meta_path))
        if ref_frame_mode not in ("window_first", "random"):
            raise ValueError(f"unknown ref_frame_mode: {ref_frame_mode}")
        self.ref_frame_mode = ref_frame_mode
        if self.target_hw_buckets is None:
            raise ValueError("RenderVideoDataset needs target_hw buckets")
        # Only needed when the metadata predates render_lag_frames.
        path = offsets_path or os.path.join(self.basedir_name, "clip_offsets.jsonl")
        if any(r.get("render_lag_frames") is not None for r in self.vid_meta[:64]):
            self.clip_offsets, self.default_offset = {}, 0
            n_meas = sum(1 for r in self.vid_meta if r.get("render_lag_measured"))
            print(f"[render-data] render lag from metadata: "
                  f"{n_meas}/{len(self.vid_meta)} measured, rest at the corpus median")
        else:
            self.clip_offsets, self.default_offset = self._load_offsets(path)

    @staticmethod
    def _load_offsets(path):
        """uuid -> render lag in frames at 24 fps, plus the fallback median.

        Only entries whose peak is prominent and whose two independent scores
        agree are trusted; the rest fall back to the median of the trusted set.
        """
        import json
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run Render/scripts/measure_offsets.py first; "
                f"pairing without it feeds the model a control signal ~0.6 s early.")
        table, trusted = {}, []
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("ok"):
                    continue
                if r["prominence"] > 1.0 and abs(r["offset_spatial"] - r["offset_motion"]) <= 3:
                    table[r["uuid"]] = int(r["offset"])
                    trusted.append(int(r["offset"]))
        if not trusted:
            raise ValueError(f"{path} has no confidently measured clip")
        median = int(np.median(trusted))
        print(f"[render-data] offsets: {len(table)} measured, median {median:+d} used elsewhere")
        return table, median

    def _bucket_geometry(self, ori_h, ori_w):
        target_h, target_w = self._map_to_nearest_bucket(ori_h, ori_w)
        target_h = max(16, int(round(target_h / 16.0) * 16))
        target_w = max(16, int(round(target_w / 16.0) * 16))
        scale_ratio = max(target_h / float(ori_h), target_w / float(ori_w))
        h = int(round(ori_h * scale_ratio))
        w = int(round(ori_w * scale_ratio))
        if h < target_h or w < target_w:
            return None
        return h, w, target_h, target_w, (h - target_h) // 2, (w - target_w) // 2

    # Render/scripts/infer.py requests 5.0 s, the model rounds its output length
    # up to 4n+4 (124 frames / 5.167 s at 24 fps), and align_frames.py cuts that
    # back to the 120 frames the input holds -- "truncating makes the two
    # frame-aligned, which is what makes the pair usable as paired data". The
    # published dataset ships the untruncated renders, so that cut happens here.
    REFERENCE_SECONDS = 5.0

    @classmethod
    def _to_reference_indices(cls, times, ref_fps, ref_vlen, offset=0):
        # Both sides are indexed by absolute time from the clip start, and the
        # render additionally lags by `offset` frames at 24 fps, so a real frame
        # at time t pairs with render frame round(t * 24) + offset.
        usable = min(ref_vlen, int(round(cls.REFERENCE_SECONDS * ref_fps)) + offset)
        idx = np.round(times * ref_fps).astype(np.int64) + offset
        return np.clip(idx, 0, usable - 1)

    @staticmethod
    def _to_video(frames, crop_top, crop_left, new_h, new_w):
        frames = frames if torch.is_tensor(frames) else torch.from_numpy(frames)
        frames = frames[:, crop_top:crop_top + new_h, crop_left:crop_left + new_w]
        frames = frames.permute(0, 3, 1, 2).to(torch.float32)
        return ((frames - 127.5) / 127.5).to(torch.bfloat16)

    @classmethod
    def make_datasets(cls, paths, args, infer=False, dataset_scale=1, **kwargs):
        # The parent hardcodes RandomMappingDataset(scale=1). That wrapper maps
        # each index to a clip drawn *with replacement*, so scale=1 leaves only
        # ~63% of the clips reachable at all, and always the same ones. Scaling
        # the index space up fixes the coverage and, since the window is now a
        # deterministic function of the index, also yields several distinct
        # windows per clip.
        from .data_v2v import ConcatDataset, MappingDataset, RandomMappingDataset

        weights = list(args.dataset_weights) if getattr(args, "dataset_weights", None) is not None else None
        datasets = [cls.create_dataset_function(p, args, **kwargs) for p in paths]
        concat = ConcatDataset(datasets, weights=weights)
        wrapper = MappingDataset if infer else RandomMappingDataset
        return wrapper(concat, scale=dataset_scale)

    # ConcatDataset only forwards the draw index to datasets that advertise this.
    accepts_draw_seed = True

    def __getitem__(self, index):
        # (clip_index, draw_index). The clip picks the file, the draw picks the
        # window: seeding the window on the clip alone made every draw of a clip
        # return the identical crop, so dataset_scale bought coverage but not a
        # single extra frame of content.
        draw = index
        if isinstance(index, tuple):
            index, draw = index
        decord.bridge.set_bridge("torch")
        meta = self.vid_meta[index]

        video_path = self._resolve_path(meta.get("video", None))
        ref_video_path = self._resolve_path(meta.get("reference_video", None))
        if video_path is None or ref_video_path is None:
            return None
        if not (os.path.exists(video_path) and os.path.exists(ref_video_path)):
            return None

        caption = (meta.get("caption") or "").strip()
        if caption == "":
            # The caption is the only thing carrying appearance here, so a clip
            # without one has nothing to condition on.
            return None

        try:
            vr_probe = VideoReader(uri=video_path)
            ori_vlen = len(vr_probe)
            fps_val = float(vr_probe.get_avg_fps()) if hasattr(vr_probe, "get_avg_fps") else 24.0
            fr0 = vr_probe[0]
        except Exception:
            return None
        ori_h, ori_w = int(fr0.shape[0]), int(fr0.shape[1])

        if ori_vlen < self.max_num_frames:
            return None

        if fps_val > 0 and self.target_fps > 0:
            # Fractional resampling, unlike the parent's integer ceil() stride.
            # The stride version maps 29.97 -> step 2 -> 15 fps, halving the
            # temporal resolution of the bulk of this corpus. Landing every clip
            # on 24 fps reproduces what prepare_inputs.py fed the renderer, which
            # is what makes frame k here and render frame k the same instant.
            n_out = int(np.floor((ori_vlen - 1) * self.target_fps / fps_val)) + 1
            sampled_indices = np.round(
                np.arange(n_out) * fps_val / self.target_fps
            ).astype(np.int64)
            sampled_indices = np.clip(sampled_indices, 0, ori_vlen - 1)
            frame_fps = self.target_fps
        else:
            sampled_indices = np.arange(ori_vlen, dtype=np.int64)
            frame_fps = fps_val

        # Crop to the window the render actually covers. Anything past
        # REFERENCE_SECONDS has no counterpart on the reference side.
        if frame_fps > 0:
            keep = int(round(self.REFERENCE_SECONDS * frame_fps))
            sampled_indices = sampled_indices[:keep]
        sampled_vlen = len(sampled_indices)
        if sampled_vlen < max(self.target_len, self.motion_len):
            return None

        geometry = self._bucket_geometry(ori_h, ori_w)
        if geometry is None:
            return None
        h, w, new_h, new_w, crop_top, crop_left = geometry

        try:
            vr = VideoReader(uri=video_path, height=h, width=w)
            vr_ref = VideoReader(uri=ref_video_path, height=h, width=w)
        except Exception:
            return None
        ref_vlen = len(vr_ref)

        # Seeded on the index so the window this returns is a function of the
        # index alone. TokenBucketSampler validates an index by calling
        # getitem_fast in the main process, and the DataLoader worker then runs
        # __getitem__ again; if the two draws differed, a decode that only fails
        # for one of them would return None to RandomMappingDataset.__getitem__
        # and blow up on `data_buffer.pop(index)`.
        # Applying the render's lag costs the tail of the real clip: real frame t
        # needs render frame t + offset, so anything past the render's last frame
        # has no counterpart and would silently freeze the control signal on the
        # final render frame. Draw the window from the part that is actually
        # covered rather than clipping after the fact.
        # The measured lag is baked into the metadata (render_lag_frames, written
        # by Render/scripts/measure_offsets.py), so the dataset is self-contained;
        # the sidecar is only a fallback for metadata written before that.
        offset = meta.get("render_lag_frames")
        if offset is None:
            offset = self.clip_offsets.get(meta.get("uuid"), self.default_offset)
        # Clamp to >= 0. The render is generated *from* the source, so it cannot
        # lead it; a negative measurement is estimator noise. Left signed, the
        # early frames of a clip resolve to a negative render index, get clipped
        # to 0, and silently repeat render frame 0 -- the same frozen-control-
        # signal failure the tail is guarded against below. 203 of 9773 clips
        # measure negative, mostly -1/-2, but a dozen sit at -14..-60 where
        # nearly every window would have been affected.
        offset = max(0, int(offset))
        ref_fps = float(meta.get("reference_video_fps", 24.0) or 24.0)
        max_ref = min(ref_vlen, int(round(self.REFERENCE_SECONDS * ref_fps)) + offset)
        covered = min(sampled_vlen, max(0, max_ref - offset))
        if covered < self.target_len:
            return None

        rng = random.Random(draw)
        target_start = rng.randint(0, covered - self.target_len)
        target_end = target_start + self.target_len
        target_indices = np.arange(target_start, target_end, dtype=np.int64)
        motion_indices = np.array(rng.sample(range(covered), self.motion_len), dtype=np.int64)

        orig_target = sampled_indices[target_indices]
        orig_motion = sampled_indices[motion_indices]
        wanted = np.concatenate([orig_target, orig_motion])
        # Positions are on the resampled timeline, so their timestamps are
        # position / frame_fps; at target_fps == 24 this is index identity
        # against the render.
        positions = np.concatenate([target_indices, motion_indices])
        wanted_ref = self._to_reference_indices(
            positions.astype(np.float64) / max(frame_fps, 1e-6), ref_fps, ref_vlen, offset
        )

        try:
            frms = vr.get_batch(wanted)
            frms_ref = vr_ref.get_batch(wanted_ref)
            if self.ref_frame_mode == "random":
                ref_frame = vr[rng.randint(0, ori_vlen - 1)].unsqueeze(0)
            else:
                ref_frame = frms[:1]
        except Exception:
            return None

        n_target = len(orig_target)
        tensor_frms = self._to_video(frms[:n_target], crop_top, crop_left, new_h, new_w)
        tensor_frms_pose = self._to_video(frms_ref[:n_target], crop_top, crop_left, new_h, new_w)
        motion_frms = self._to_video(frms[n_target:], crop_top, crop_left, new_h, new_w)
        motion_frms_pose = self._to_video(frms_ref[n_target:], crop_top, crop_left, new_h, new_w)
        tensor_ref = self._to_video(ref_frame, crop_top, crop_left, new_h, new_w)

        cond_h, cond_w = new_h // self.lat_mul_size, new_w // self.lat_mul_size
        f_lat = (self.target_len - 1) // 4 + 1
        token_num = float(cond_h * cond_w * f_lat)

        item = {
            'item_name': meta.get("uuid", os.path.splitext(os.path.basename(video_path))[0]),
            'video_path': video_path,
            'start': int(target_start),
            'end': int(target_end),
            "tgt_frame": tensor_frms,
            "tgt_frame_pose": tensor_frms_pose,
            "text": caption,
            "fps": int(round(self.target_fps)) if self.target_fps > 0 else int(round(fps_val)),
            "ref_image": tensor_ref,
            "token_num": token_num,
        }
        if self.enable_motion_frames:
            item["motion_frame"] = motion_frms
            item["motion_frame_pose"] = motion_frms_pose
        return item
