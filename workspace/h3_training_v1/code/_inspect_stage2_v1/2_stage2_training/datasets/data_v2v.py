import os
import random
import cv2
import copy
from einops import rearrange
import orjson
import torch
import torchvision
from torchvision.transforms import v2
import numpy as np
import time, threading
from PIL import Image
from torch.utils.data import Sampler
from glob import glob
from bisect import bisect_right
import decord
from decord import VideoReader
import torch.distributed as dist
import yaml

def cycle(dl):
    while True:
        for data in dl:
            yield data

def fast_read_jsonl(file):
    all_data = []
    with open(file, "rb") as f:
        for line in f:
            all_data.append(orjson.loads(line))  
    return all_data  

class AnimateVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_meta_path, 
                 max_num_frames=153, 
                 dataset_max_token=20000,
                 max_token=20000, 
                 max_wh=720, 
                 enable_face_augment=True,
                 enable_motion_frames=False,
                 target_hw=None,
                 target_len=33,
                 motion_len=17,
                 target_fps=16.0
                ):
        self.basedir_name = data_meta_path
        self.max_num_frames = max_num_frames
        self.max_wh = max_wh
        self.target_hw = target_hw
        
        vid_meta = fast_read_jsonl(data_meta_path)
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        # print(f'load dataset {os.path.basename(data_meta_path)}, len {self.length}')
        
        self.lat_mul_size = 8
        self.max_token = max_token
        self.dataset_max_token = dataset_max_token

        self.enable_face_augment = enable_face_augment
        self.enable_motion_frames = enable_motion_frames
        self.target_len = target_len
        self.motion_len = motion_len
        self.target_fps = target_fps
        if self.enable_motion_frames:
            self.motion_frames = [self.motion_len,(self.motion_len-1)//4+1]
        
        if self.enable_face_augment:
            print("!!augment_transform is used!!!")
            # self.augment_transform = v2.Compose([
            #     v2.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # scaling
            #     v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            #     v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.5),
            # ])
            self.augment_transform = v2.Compose([
                v2.RandomApply([v2.ElasticTransform(alpha=200.0, sigma=6.0)], p=1),
                v2.RandomAffine(degrees=0, scale=(0.7, 1.3), translate=(0.1, 0.1)),
                v2.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3),
                v2.RandomApply([
                    v2.GaussianBlur(kernel_size=27, sigma=(0.4, 0.8))
                ], p=1),
            ])
        else:
            self.augment_transform = None

        # bucket list for target_hw
        self.target_hw_buckets = None
        if self.target_hw is not None:
            try:
                hw_list = list(self.target_hw)
                if len(hw_list) > 0 and hasattr(hw_list[0], '__iter__') and not isinstance(hw_list[0], str):
                    self.target_hw_buckets = [(int(h), int(w)) for h, w in hw_list]
                elif len(hw_list) == 2 and isinstance(hw_list[0], int):
                    self.target_hw_buckets = [(int(hw_list[0]), int(hw_list[1]))]
            except Exception:
                pass
    
    def _resolve_path(self, p: str):
        if p is None:
            return None
        if isinstance(p, str) and (p.startswith("http://") or p.startswith("https://")):
            return p
        if isinstance(p, str) and os.path.isabs(p):
            return p
        return os.path.join(self.basedir_name, p)

    def _map_to_nearest_bucket(self, ori_h: int, ori_w: int):
        src_log_ratio = np.log(max(float(ori_w) / float(max(ori_h, 1)), 1e-8))
        best_bucket, best_dist = self.target_hw_buckets[0], float("inf")
        for bh, bw in self.target_hw_buckets:
            dist = abs(np.log(max(float(bw) / float(max(bh, 1)), 1e-8)) - src_log_ratio)
            if dist < best_dist:
                best_dist = dist
                best_bucket = (bh, bw)
        return best_bucket
    
    def __getitem__(self, index):
        decord.bridge.set_bridge("torch")
        meta = self.vid_meta[index]

        video_path = self._resolve_path(meta.get("video", None))
        pose_video_path = self._resolve_path(meta.get("pose", None))
        face_video_path = self._resolve_path(meta.get("face", None))
        ref_path = self._resolve_path(meta.get("ref", None))
        caption_path = self._resolve_path(meta.get("caption", None))

        if (not os.path.exists(pose_video_path)) or (not os.path.exists(face_video_path)):
            # print(f'| missing pose or face video for {video_path}')
            return None

        # 读取 caption 与 caption embedding
        caption = None
        captionemb = None
        if caption_path is not None and os.path.exists(caption_path):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            except Exception:
                caption = None
        if (caption is None or caption == ""):
            caption = 'A realistic video of a human talking directly to the camera, with natural facial expressions.'
        
        # 新格式通常没有 caption embedding
        
        vr_probe = VideoReader(uri=video_path)
        ori_vlen = len(vr_probe)
        fps_val = float(vr_probe.get_avg_fps()) if hasattr(vr_probe, "get_avg_fps") else 25.0
        # print(f"Video {video_path} has {ori_vlen} frames at {fps_val:.2f} fps.")
        fr0 = vr_probe[0]
        ori_h, ori_w = int(fr0.shape[0]), int(fr0.shape[1])
                
        if ori_vlen < self.max_num_frames:
            return None
        
        # 目标采样 fps
        if fps_val > 0 and self.target_fps > 0:
            sample_step = max(1, int(np.ceil(fps_val / self.target_fps)))
        else:
            sample_step = 1

        # 在原视频索引上做重采样
        sampled_indices = np.arange(0, ori_vlen, sample_step, dtype=np.int64)
        sampled_vlen = len(sampled_indices)
        if sampled_vlen < max(self.target_len, self.motion_len):
            return None

        crop_top, crop_left = 0, 0
        if self.target_hw_buckets is None:
            # 旧逻辑：max_wh 缩放 + 16 对齐中心裁
            if self.max_wh is not None and self.max_wh > 0:
                scale_ratio = min(1.0, float(self.max_wh) / float(max(ori_h, ori_w)))
            else:
                scale_ratio = 1.0
            h = int(round(ori_h * scale_ratio))
            w = int(round(ori_w * scale_ratio))
            new_h = h - (h % 16)
            new_w = w - (w % 16)
            if new_h <= 0 or new_w <= 0:
                return None
            crop_top, crop_left = (h - new_h) // 2, (w - new_w) // 2
        else:
            # bucket 模式：宽高比最近邻映射 -> scale -> 中心裁剪（与 stage1 完全一致）
            target_h, target_w = self._map_to_nearest_bucket(ori_h, ori_w)
            target_h = max(16, int(round(target_h / 16.0) * 16))
            target_w = max(16, int(round(target_w / 16.0) * 16))
            scale_ratio = max(target_h / float(ori_h), target_w / float(ori_w))
            h = int(round(ori_h * scale_ratio))
            w = int(round(ori_w * scale_ratio))
            if h < target_h or w < target_w:
                return None
            crop_top = (h - target_h) // 2
            crop_left = (w - target_w) // 2
            new_h, new_w = target_h, target_w

        # 用统一的 h,w 创建 reader（face 也要一致）
        try:
            vr = VideoReader(uri=video_path, height=h, width=w)
            vr_pose = VideoReader(uri=pose_video_path, height=h, width=w)
            vr_face = VideoReader(uri=face_video_path)
        except Exception:
            return None
        
         # target/pose/face：在“重采样后序列”上取连续帧
        target_start = random.randint(0, sampled_vlen - self.target_len)
        target_end = target_start + self.target_len
        target_indices = np.arange(target_start, target_end, dtype=np.int64)
        
        # motion：全局随机17帧
        motion_indices = np.array(random.sample(range(sampled_vlen), self.motion_len), dtype=np.int64)

        # ref：全局随机1帧
        ref_idx = random.randint(0, ori_vlen - 1)

        # 全读（你要求的方式）
        try:
            all_indices = np.arange(ori_vlen, dtype=np.int64)
            temp_frms_all = vr.get_batch(all_indices)
            temp_frms_pose_all = vr_pose.get_batch(all_indices)
            temp_frms_face_all = vr_face.get_batch(all_indices)
        except Exception:
            return None

        # 中心裁剪区域
        temp_frms_all = temp_frms_all[:, crop_top:crop_top + new_h, crop_left:crop_left + new_w]
        temp_frms_pose_all = temp_frms_pose_all[:, crop_top:crop_top + new_h, crop_left:crop_left + new_w]

        # 取 target 序列
        tensor_frms = temp_frms_all[target_indices]
        tensor_frms_pose = temp_frms_pose_all[target_indices]
        tensor_frms_face = temp_frms_face_all[target_indices]

        tensor_frms = torch.from_numpy(tensor_frms) if type(tensor_frms) is not torch.Tensor else tensor_frms
        tensor_frms_pose = torch.from_numpy(tensor_frms_pose) if type(tensor_frms_pose) is not torch.Tensor else tensor_frms_pose
        tensor_frms_face = torch.from_numpy(tensor_frms_face) if type(tensor_frms_face) is not torch.Tensor else tensor_frms_face

        # ref 帧
        ref_image = temp_frms_all[ref_idx]
        tensor_ref = torch.from_numpy(ref_image) if type(ref_image) is not torch.Tensor else ref_image

        # 预处理视频帧/参考帧/条件帧
        tensor_ref = tensor_ref.permute(2, 0, 1).unsqueeze(0)
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)
        tensor_frms_pose = tensor_frms_pose.permute(0, 3, 1, 2)
        tensor_frms_face = tensor_frms_face.permute(0, 3, 1, 2)

        tensor_frms = (tensor_frms - 127.5) / 127.5
        tensor_frms_pose = (tensor_frms_pose - 127.5) / 127.5
        tensor_frms_face = (tensor_frms_face - 127.5) / 127.5
        tensor_ref = (tensor_ref - 127.5) / 127.5

        # Face augmentation
        if self.enable_face_augment and self.augment_transform is not None:
            tensor_frms_face = (tensor_frms_face + 1) / 2
            tensor_frms_face = self.augment_transform(tensor_frms_face)
            tensor_frms_face = tensor_frms_face * 2 - 1

        # motion_frame（随机17帧）
        if self.enable_motion_frames:
            motion_frms = temp_frms_all[motion_indices]
            motion_frms_pose = temp_frms_pose_all[motion_indices]

            motion_frms = torch.from_numpy(motion_frms) if type(motion_frms) is not torch.Tensor else motion_frms
            motion_frms_pose = torch.from_numpy(motion_frms_pose) if type(motion_frms_pose) is not torch.Tensor else motion_frms_pose

            motion_frms = motion_frms.permute(0, 3, 1, 2)
            motion_frms_pose = motion_frms_pose.permute(0, 3, 1, 2)

            motion_frms = (motion_frms - 127.5) / 127.5
            motion_frms_pose = (motion_frms_pose - 127.5) / 127.5


        # token_num：无 latent 时给近似值（供 sampler）
        cond_h, cond_w = new_h // self.lat_mul_size, new_w // self.lat_mul_size
        f_lat = (self.target_len - 1) // 4 + 1
        token_num = float(cond_h * cond_w * f_lat)

        item = {
            'item_name': os.path.basename(os.path.dirname(video_path)) or os.path.splitext(os.path.basename(video_path))[0],
            'video_path': video_path,
            'start': int(target_start),
            'end': int(target_end),
            "tgt_frame": tensor_frms.to(torch.bfloat16),
            "tgt_frame_pose": tensor_frms_pose.to(torch.bfloat16),
            "tgt_frame_face": tensor_frms_face.to(torch.bfloat16),
            "text": caption,
            "fps": (int(round(fps_val)) // sample_step) if fps_val > 0 else 0,
            "ref_image": tensor_ref.to(torch.bfloat16),
            "token_num": token_num,
        }

        if self.enable_motion_frames:
            item["motion_frame"] = motion_frms.to(torch.bfloat16)
            item["motion_frame_pose"] = motion_frms_pose.to(torch.bfloat16)

        if captionemb is not None:
            item['captionemb'] = captionemb

        return item

    def __len__(self):
        return len(self.vid_meta)
    
    # 工厂方法：给一个 path 创建一个 AudioVideoDataset 实例
    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_meta_path=path, **kwargs)
    
    @classmethod
    def make_datasets(cls, paths, args, infer=False, **kwargs):
        ds = []
        weights = [] if getattr(args, 'dataset_weights', None) is not None else None
        for i, p in enumerate(paths):
            d = cls.create_dataset_function(p, args, **kwargs)
            ds.append(d)
            if weights is not None:
                weights.append(args.dataset_weights[i])
            if getattr(args, 'dataset_mix_wh', False): # 如果配置里 dataset_mix_wh=True（only used for add 1280 to 720），实际上config为False
                new_kwargs = copy.deepcopy(kwargs)
                new_kwargs['meta_name'] = new_kwargs['meta_name'].replace('720', '1280')
                new_kwargs['dataset_max_token'] = 120000
                new_kwargs['max_wh'] = 1280
                d = cls.create_dataset_function(p, args, **new_kwargs)
                ds.append(d)
                if weights is not None:
                    weights.append(args.dataset_weights[i])
        ds = ConcatDataset(ds, weights=weights) # 把多个 dataset 合并成一个逻辑上的“大数据集”，同时用权重控制每个子数据集的长度
        if infer:
            ds = MappingDataset(ds, scale=1) # 把一个 dataset “虚拟扩容”为 scale 倍长度，并提供快速缓存
        else:
            ds = RandomMappingDataset(ds, scale=1) # 与 MappingDataset 类似，但不是 index % len(ds)，而是根据 index 生成一个随机种子，从底层随机抽一个 new_index，保证确定性。
        return ds
    

class ConcatDataset(torch.utils.data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:  
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence, weights):
        r, s = [], 0
        for i, e in enumerate(sequence):
            l = int(len(e) * weights[i])
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, weights=None, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        if weights is None:
            self.weights = [1] * len(self.datasets)
        else:
            self.weights = weights
        self.cumulative_sizes = self.cumsum(self.datasets, self.weights)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # RandomMappingDataset draws a clip from its own enlarged index space, so
        # only it knows which *draw* this is. A dataset that varies the crop per
        # draw needs that number; without it the crop is a function of the clip
        # alone and every draw of the same clip returns the identical sample.
        draw_seed = None
        if isinstance(idx, tuple):
            idx, draw_seed = idx
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % len(self.datasets[dataset_idx])
        ds = self.datasets[dataset_idx]
        if draw_seed is not None and getattr(ds, "accepts_draw_seed", False):
            return ds[(sample_idx, draw_seed)]
        return ds[sample_idx]
    
class MappingDataset(torch.utils.data.Dataset):
    '''
    Dataset wrapper to randomly mapping indices to original order.
    Will also enlarge the length
    '''
    def __init__(self, ds, scale=200, **kwargs):
        self.wrapped_data = ds
        self.scale = scale
        self.data_buffer = {}

    def __len__(self):
        return len(self.wrapped_data) * self.scale
    
    def getitem_fast(self, index):
        new_index = index % len(self.wrapped_data)
        item = self.wrapped_data[new_index]
        if item is not None:
            self.data_buffer[index] = item
        return item

    def __getitem__(self, index):
        if index not in self.data_buffer.keys():
            self.getitem_fast(index)
        return self.data_buffer.pop(index)

class RandomMappingDataset(torch.utils.data.Dataset):
    '''
    Dataset wrapper to randomly mapping indices to original order.
    Will also enlarge the length
    '''
    def __init__(self, ds, scale=200, **kwargs):
        self.wrapped_data = ds
        self.scale = scale
        self.data_buffer = {}

    def __len__(self):
        return len(self.wrapped_data) * self.scale
    
    def getitem_fast(self, index):
        rng = random.Random(index)
        rng = np.random.RandomState(seed=[rng.randint(0, 2**32-1) for _ in range(16)])
        new_index = rng.randint(len(self.wrapped_data))
        item = self.wrapped_data[(new_index, index)]
        if item is not None:
            self.data_buffer[index] = item
        return item

    def __getitem__(self, index):
        if index not in self.data_buffer.keys():
            self.getitem_fast(index)
        return self.data_buffer.pop(index)
    
# 自定义 Sampler，用“token bucket”策略：根据 latent token 数（"token_num"）来选样本，使不同 GPU 上选到的样本 token 数尽量相近。
class TokenBucketSampler(Sampler):
    """
    与 data_v2v.py 完全一致的 TokenBucketSampler，可直接复用。
    """

    def __init__(self, dataset, batch_size=1, preload_size=128, world_size=1, rank=0, device="cuda"):
        self.dataset = dataset
        self.preload_size = preload_size
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.batch_size = batch_size

        self.buffer_indices = []
        self.buffer_tokens = []

        # 每个 rank 只负责数据集的一个切片
        per_rank = len(self.dataset) // self.world_size
        start = per_rank * self.rank
        end = per_rank * (self.rank + 1) if self.rank < self.world_size - 1 else len(self.dataset)
        self.rank_indices = list(range(start, end))

        self.idx_pointer = 0
        self.shuffle()
        self.lock = threading.Lock()

        self._stop_flag = False
        self.thread = threading.Thread(target=self._preload_thread)
        self.thread.daemon = True
        self.thread.start()

    def shuffle(self):
        random.shuffle(self.rank_indices)
        self.idx_pointer = 0

    def _next_index(self):
        idx = self.rank_indices[self.idx_pointer]
        self.idx_pointer += 1
        if self.idx_pointer >= len(self.rank_indices):
            self.shuffle()
        return idx

    def preload(self):
        while len(self.buffer_indices) < self.preload_size // 5:
            idx = self._next_index()
            sample = self.dataset.getitem_fast(idx)
            if sample is not None:
                token = sample["token_num"]
                self.buffer_indices.append(idx)
                self.buffer_tokens.append(token)

    def _preload_thread(self):
        while not self._stop_flag:
            with self.lock:
                if len(self.buffer_indices) >= self.preload_size:
                    pass
                else:
                    idx = self._next_index()
                    sample = self.dataset.getitem_fast(idx)
                    if sample is not None:
                        token = sample["token_num"]
                        self.buffer_indices.append(idx)
                        self.buffer_tokens.append(token)
            time.sleep(0.01)

    def __iter__(self):
        while True:
            while True:
                with self.lock:
                    if len(self.buffer_indices) > 1:
                        break
                time.sleep(0.01)

            if self.rank == 0:
                target_token = random.choice(self.buffer_tokens)
                token_tensor = torch.tensor([target_token], device=self.device, dtype=torch.long)
            else:
                token_tensor = torch.tensor([0], device=self.device, dtype=torch.long)

            dist.broadcast(token_tensor, src=0)
            target_token = token_tensor.item()

            with self.lock:
                diffs = [abs(t - target_token) for t in self.buffer_tokens]
                min_idx = diffs.index(min(diffs))

                chosen_idx = self.buffer_indices[min_idx]
                del self.buffer_indices[min_idx]
                del self.buffer_tokens[min_idx]
            yield [chosen_idx]
            
def prepare_multimodal_batch(batch,device,dtype,use_usp=False,use_captionemb=False,args=None,neg_cond_prompt=None):
    batch_size = batch["tgt_frame"].shape[0]
    if use_usp:
        print("USP not implemented.")
        pass
    
    cond_dict = {}
    if use_captionemb:
        cond_dict['context'] = batch['captionemb']
        cond_dict["context"] = cond_dict["context"].to(device).to(dtype=dtype)
    else:
        cond_dict['prompt'] = batch['text']
        
    cond_dict["ref_image"] = batch["ref_image"].squeeze(0).to(device).to(dtype=dtype) # shape: (B, C, H, W)
    cond_dict["face"] = batch["tgt_frame_face"].to(device).to(dtype=dtype) 
    cond_dict["face"] = rearrange(cond_dict["face"], 'b t c h w -> b c t h w') # shape: (B, C, T, H, W)
    cond_dict["pose"] = batch["tgt_frame_pose"].to(device).to(dtype=dtype) 
    cond_dict["pose"] = rearrange(cond_dict["pose"], 'b t c h w -> b c t h w') # shape: (B, C, T, H, W)
    b, _, F, H, W = cond_dict["pose"].shape
    
    neg_cond_dict = {'neg_flag': True}
    neg_cond_dict['prompt'] = [neg_cond_prompt] * batch_size
    
    image_or_video_shape = list(cond_dict["pose"].shape) # b, c, F_lat, H_lat, W_lat
    image_or_video_shape = [image_or_video_shape[0], image_or_video_shape[2]//4, 16, image_or_video_shape[3] // 8, image_or_video_shape[4] // 8] # b, 16, F_lat//4, H_lat, W_lat
    return batch_size, cond_dict, neg_cond_dict, image_or_video_shape


if __name__ == '__main__':

    def tensor_img_to_uint8(chw_tensor: torch.Tensor):
        """
        输入: [C, H, W]，范围约在 [-1, 1]
        输出: HWC uint8 图像
        """
        t = chw_tensor.to(torch.float32)
        t = (t * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        t = t.permute(1, 2, 0).cpu().numpy()  # HWC
        return t

    def save_tensor_video(tchw: torch.Tensor, out_path: str, fps: int = 25):
        # tchw: [T, C, H, W], 值域约[-1,1]
        tchw = tchw.to(torch.float32).cpu()
        T, C, H, W = tchw.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, max(1, int(fps)), (W, H))
        for i in range(T):
            rgb = tensor_img_to_uint8(tchw[i])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

    def visualize_sample(sample: dict, output_dir: str, index: int = 0):
        os.makedirs(output_dir, exist_ok=True)

        # 1) 打印并保存 caption
        caption = sample.get("text", "")
        print("caption:", caption)
        with open(os.path.join(output_dir, "caption.txt"), "w", encoding="utf-8") as f:
            f.write(caption if caption is not None else "")

        # 2) 保存 ref 图片
        ref_img = tensor_img_to_uint8(sample["ref_image"][0])
        Image.fromarray(ref_img).save(os.path.join(output_dir, "ref_image.png"))

        # 3) 保存完整 target / pose / face 视频
        fps = sample.get("fps", 25)
        save_tensor_video(sample["tgt_frame"], os.path.join(output_dir, f"{index}tgt_full.mp4"), fps=fps)
        save_tensor_video(sample["tgt_frame_pose"], os.path.join(output_dir, f"{index}pose_full.mp4"), fps=fps)
        save_tensor_video(sample["tgt_frame_face"], os.path.join(output_dir, f"{index}face_full.mp4"), fps=fps)
        
        if "motion_frame_pose" in sample and sample["motion_frame_pose"] is not None:
            save_tensor_video(sample["motion_frame_pose"], os.path.join(output_dir, f"{index}motion_pose_full.mp4"), fps=fps)
#
        # 4) 保存 motion 视频（如果有）
        if "motion_frame" in sample and sample["motion_frame"] is not None:
            save_tensor_video(sample["motion_frame"], os.path.join(output_dir, f"{index}motion_full.mp4"), fps=fps)

        print(f"Saved visualizations to: {output_dir}")
            
    # 示例用法
    # ==== 这里开始真正测试 ====
    dataset = AnimateVideoDataset(
        data_meta_path='/primus_xpfs_workspace_T04/zyx/data/liveanimate/merged_3_5.jsonl',
        max_num_frames=50,
        dataset_max_token=80000,
        max_token=60000,
        max_wh=720,
        enable_face_augment=True,
        enable_motion_frames=True,
        target_hw=(720*400),
        target_len=33,
        motion_len=17,
        target_fps=16.0
    )

    print(f"Dataset length: {len(dataset)}")

    save_dir = "/primus_xpfs_workspace_T04/zyx/projects/project2/LA_stage1_0315_motion/diffsynth/core/data/tmp_vis"
    os.makedirs(save_dir, exist_ok=True)
    # ✅ 直接用 dataset[i]，一定会走 __getitem__
    for idx in range(len(dataset)):
        print(f"trying index {idx}")
        sample = dataset[idx]          # 这里就是在调用 __getitem__
        if sample is None:
            print(f"Sample {idx} is None, skip.")
            continue
        visualize_sample(sample, save_dir, index=idx)
        
        # ======== 维度检查部分开始 ========
        tgt = sample["tgt_frame"]          # [T, C, H, W]
        tgt_pose = sample["tgt_frame_pose"]# [T, C, H, W]
        motion = sample.get("motion_frame", None)  # [T_motion, C, H, W] 或 None
        face = sample["tgt_frame_face"]    # [T, C, H, W]
        fps = sample.get("fps", 25)
        motion_pose = sample.get("motion_frame_pose", None)  # [T_motion, C, H, W] 或 None
        
        T_img = tgt.shape[0]
        T_pose = tgt_pose.shape[0]

        H_img, W_img = tgt.shape[-2], tgt.shape[-1]

        print("=== shape info ===")
        print(f"tgt_frame      : {tgt.shape}")       # [T, C, H, W]
        print(f"tgt_frame_pose : {tgt_pose.shape}")  # [T, C, H, W]
        print(f"tgt_frame_face : {face.shape}")      # [T, C, H, W]
        print(f"fps            : {fps}")
        if motion is not None:
            print(f"motion_frame   : {motion.shape}")    # [T_motion, C, H, W]
            print(f"motion_frame_pose : {motion_pose.shape}")  # [T_motion, C, H, W]

        # 时间维：按当前写法，理论关系是 T = 4*(F_lat - 1) + 1
        print(f"T_img={T_img}, T_pose={T_pose}, ")

        if T_img != T_pose:
            print("[WARN] tgt_frame 和 tgt_frame_pose 的时间长度不一致！")

        # 理论上这里接近 8（lat_mul_size）或 16，看预处理配置
        # ======== 维度检查部分结束 ========

        # 如需继续保存可视化可以保留这行
        # visualize_sample(sample, save_dir)

        # 做完一次检查就退出
        if idx >= 4:
            break
    else:
        print("No valid sample found.")