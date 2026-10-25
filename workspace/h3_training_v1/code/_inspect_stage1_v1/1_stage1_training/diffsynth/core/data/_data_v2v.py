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
                 meta_name='v1_total', 
                 dataset_max_token=20000,
                 max_token=20000, 
                 max_wh=720, 
                 frame_interval=1, 
                 caption_mode='tarsier', 
                 force_captionemb=True,
                 enable_face_augment=True,
                 enable_motion_frames=False,
                ):
        self.basedir_name = data_meta_path
        self.max_num_frames = max_num_frames
        self.max_wh = max_wh
        self.frame_interval = frame_interval

        vid_meta = fast_read_jsonl( f'{data_meta_path}/meta_dataset_{meta_name}.jsonl')
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        print(f'load dataset {os.path.basename(data_meta_path)}, meta name {meta_name}, len {self.length}')
        
        self.lat_mul_size = 8
        self.max_token = max_token
        self.dataset_max_token = dataset_max_token

        self.caption_mode = caption_mode
        self.force_captionemb = force_captionemb
        self.enable_face_augment = enable_face_augment
        self.enable_motion_frames = enable_motion_frames
        if self.enable_motion_frames:
            self.motion_frames = [13,(13-1)//4+1]
        
        if self.enable_face_augment:
            self.augment_transform = v2.Compose([
                v2.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # scaling
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.5),
            ])
        else:
            self.augment_transform = None
    
    def __getitem__(self, index):
        decord.bridge.set_bridge("torch")
        video_meta = self.vid_meta[index]

        # 解析 video_meta
        item_name, video_meta = next(iter(video_meta.items())) # video_meta 本身是 {item_name: meta_dict}，通过 next(iter(...)) 拆出键值
        video_path = f'{self.basedir_name}/{video_meta["video_path"]}' # 视频完整路径
        fps = video_meta['fps'] # 视频 fps
        feature_dir = video_path[:-4] + '_features' # 同名目录 _features（里面存 caption、音频、ocr、latent 等）
        pose_video_path = video_path[:-4] + '_pose.mp4' # 同名目录 _pose.mp4（人体关键点视频路径）
        face_video_path = video_path[:-4] + '_face.mp4' # 同名目录 _face.mp4（人脸关键点视频路径）
        
        if (not os.path.exists(pose_video_path)) or (not os.path.exists(face_video_path)):
            # print(f'| missing pose or face video for {video_path}')
            return None
        ori_vlen = video_meta['frame_num'] 
        if ori_vlen < self.max_num_frames: # 121等于num_frames+motion_frames即 num_frames+73, 原视频帧数小于要求的最大帧数，则该样本直接丢弃
            # print(f'| video too short {video_path}, ori_vlen {ori_vlen}')
            return None

        # 读取 caption 与 caption embedding
        caption_yaml = f'{feature_dir}/caption.yaml'
        caption_npz = f'{feature_dir}/caption.npz'
        captionemb = None
        if os.path.exists(caption_yaml):
            with open(caption_yaml, 'r', encoding='utf-8') as file:
                caption_data = yaml.safe_load(file)  
                caption = caption_data.get(self.caption_mode, "")
            try:
                captionemb = np.zeros((512, 4096))
                captionemb_ = np.load(caption_npz, allow_pickle=True)[self.caption_mode][0]
                captionemb[:captionemb_.shape[0]] = captionemb_
            except:
                if self.force_captionemb:
                    # print(f'| error in read caption emb in {caption_npz}')
                    return None
        else:
            if self.force_captionemb:
                return None
            caption = 'A realistic video of a human talking directly to the camera, \
                with natural facial expressions, accurate lip-sync, and subtle head and body movements,\
                set against a neutral or minimal background. The speaker appears engaged and expressive, \
                as if delivering a speech or having a conversation.'
            captionemb = None

        # latent 路径与视频缩放
        ori_h, ori_w = video_meta['h'], video_meta['w']
        lat_path = video_path[:-4] + f'_lats/wan_maxwh_{self.max_wh}_maxtoken_{self.dataset_max_token}' # 720 80000
        if self.frame_interval > 1: # 1
            lat_path = lat_path + f'_stride{self.frame_interval}'
        lat_files = glob(f'{lat_path}/*.npz')
        if len(lat_files) == 0:
            # print(f'| there is no lat in {lat_path}')
            return None
        lat_file_path = random.choice(lat_files)
        try:
            lat_file = np.load(lat_file_path, allow_pickle=True)
        except:
            # print(f'| error in read lat in {lat_path}')
            return None
        lat_meta = lat_file['meta'].item()
        lat_ori = lat_file['lat'] # b, c, f, h, w
        scale_ratio = lat_meta['scale_ratio']
        clip_h, clip_w = lat_meta['hw_clip_len']
        frame_begin = lat_meta['frame_begin']
        frame_end = min(lat_meta['frame_end'], ori_vlen)

        # 读取视频与 pose、face 视频
        h, w = int(ori_h * scale_ratio), int(ori_w * scale_ratio)
        # if h == w:
        #     return None # 避免正方形视频
        vr = VideoReader(uri=video_path, height=h, width=w)
        vr_pose = VideoReader(uri=pose_video_path, height=h, width=w)
        vr_face = VideoReader(uri=face_video_path)
        
        # 计算新尺寸、token 限制、起止帧
        # clip_h, clip_w = h % 16, w % 16
        new_h, new_w = h - clip_h, w - clip_w
        cond_h, cond_w = new_h // self.lat_mul_size, new_w // self.lat_mul_size # self.lat_mul_size is 8
        max_num_frames = min(self.max_num_frames, int(self.max_token / new_h / new_w * (self.lat_mul_size**2) * 16 + 1))  # 依据 max_token 和图像尺寸计算理论上允许的视频帧上限，再和 self.max_num_frames 取 min
        # max_num_frames = max_num_frames // 4 * 4 + 1 if max_num_frames % 4 != 0 else max_num_frames - 3 # 确保max_num_frames是4n+1
        
        start_lat_idx = 0 # latent 起始 index，一般是0
        start = frame_begin # frame_begin
        end = min(start + max_num_frames * self.frame_interval, frame_end)  # end 不超过 frame_end
        ori_indices = np.arange(start, end, self.frame_interval).astype(int) # 最终要用的视频帧下标列表
        if len(ori_indices) < max_num_frames: # 如果数目不足 max_num_frames，丢样本
            # print(f'| not enough frames after interval in {video_path}, got {len(ori_indices)}, need {max_num_frames}')
            return None
        end_lat_idx = len(ori_indices) // 4 + 1 + start_lat_idx # 对应的 latent 结束 index
        
        # motion frames 前73帧
        if self.enable_motion_frames: # True self.motion_frames = 73
            motion_start = start
            motion_end = start + self.motion_frames[0] * self.frame_interval
            motion_ori_indices = np.arange(motion_start, motion_end, self.frame_interval).astype(int)
            motion_end_lat_idx = len(motion_ori_indices) // 4 + 1 + start_lat_idx  # 19 帧 motion 对应的 latent 结束 index
        
        lat = torch.from_numpy(lat_ori[:, :, start_lat_idx:end_lat_idx]) # 从原 latent 中，截取 [start_lat_idx:end_lat_idx] 时间段，转成 torch
        ori_indices = torch.from_numpy(ori_indices) # 下标转 torch
        new_indices = torch.tensor((ori_indices - start).tolist()) # 转成相对于 start 的偏移 计算相对 index (ori_indices - start)，用来在 tensor_frms 里索引对应帧
        
        # 读取完整的视频帧
        temp_frms = vr.get_batch(np.arange(start, end)) # 使用 decord 一次性读取 [[start, end) 范围的视频帧
        temp_frms = temp_frms[:, :new_h, :new_w] # 剪裁到 new_h/new_w
        assert temp_frms is not None
        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        tensor_frms = tensor_frms[new_indices] # 选取最终的视频帧 [T, H, W, C]
        
        # 读取 pose 与 face 视频帧
        try:
            temp_frms_pose = vr_pose.get_batch(np.arange(start, end))
            temp_frms_pose = temp_frms_pose[:, :new_h, :new_w]
            assert temp_frms_pose is not None
            tensor_frms_pose = torch.from_numpy(temp_frms_pose) if type(temp_frms_pose) is not torch.Tensor else temp_frms_pose
            tensor_frms_pose = tensor_frms_pose[new_indices]
            temp_frms_face = vr_face.get_batch(np.arange(start, end))
            assert temp_frms_face is not None
            tensor_frms_face = torch.from_numpy(temp_frms_face) if type(temp_frms_face) is not torch.Tensor else temp_frms_face
            tensor_frms_face = tensor_frms_face[new_indices]
        except:
            # print(f'| error in read pose or face video frames in {video_path}')
            return None
        
        # reference frame
        ref_idx = random.randint(
                0, 
                ori_vlen-1
            ) # 从整段视频帧里随机选一帧作为参考帧
        ref_idx_lat = start_lat_idx + ref_idx // 4 + 1 
        ref_idx_lat = min(ref_idx_lat, lat.shape[2]-1) # 换算到 latent 时间 index
        if ref_idx_lat < 0:  # 确保不越界
            ref_idx_lat = 0
        ref_image = vr[ref_idx] # 用 vr 单帧读取参考图像，裁剪尺寸，转为 tensor
        ref_image = ref_image[:new_h, :new_w]
        tensor_ref = torch.from_numpy(ref_image) if type(ref_image) is not torch.Tensor else ref_image
        
        # 预处理视频帧/参考帧/条件帧
        tensor_ref = tensor_ref.permute(2, 0, 1).unsqueeze(0)
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = (tensor_frms - 127.5) / 127.5
        tensor_ref = (tensor_ref - 127.5) / 127.5
        tensor_frms_face = tensor_frms_face.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms_face = (tensor_frms_face - 127.5) / 127.5
        tensor_frms_pose = tensor_frms_pose.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms_pose = (tensor_frms_pose - 127.5) / 127.5 # 归一化到 [-1, 1]
        
        # 只保留前 max_num_frames 帧（这里是截断，没有真正 pad）
        tensor_frms = tensor_frms[:max_num_frames]
        tensor_frms_face = tensor_frms_face[:max_num_frames]
        tensor_frms_pose = tensor_frms_pose[:max_num_frames]
        
        # Face augmentation: scaling, color jittering, random noise
        if self.enable_face_augment and self.augment_transform is not None:
            tensor_frms_face = (tensor_frms_face + 1) / 2 # 先还原到 [0,1] 范围
            tensor_frms_face = self.augment_transform(tensor_frms_face)
            tensor_frms_face = tensor_frms_face * 2 - 1 # 再归一化到 [-1,1]
        
        # 从 latent 时间维度上取出参考帧对应的 latent（1 帧）
        # ref_lat = lat[:, :, ref_idx_lat:ref_idx_lat+1]
        
        # 把 [start, motion_end) 的帧单独切出来做 motion_frms 与 motion_lat
        if self.enable_motion_frames:
            start = motion_end            
            motion_lat = lat[:, :, :motion_end_lat_idx]
            lat = lat[:, :, motion_end_lat_idx:]
            
            n_motion = self.motion_frames[0]
            motion_frms = tensor_frms[:n_motion]
            tensor_frms = tensor_frms[n_motion:]
            motion_frms_face = tensor_frms_face[:n_motion]
            tensor_frms_face = tensor_frms_face[n_motion:]
            motion_frms_pose = tensor_frms_pose[:n_motion]
            tensor_frms_pose = tensor_frms_pose[n_motion:]

        item = {
            'item_name': item_name,
            'video_path': video_path,
            'start': start,
            'end': end,
            "tgt_frame": tensor_frms.to(torch.bfloat16), # [T,C,H,W] 主视频帧
            "tgt_frame_pose": tensor_frms_pose.to(torch.bfloat16), # 主视频 pose 帧
            "tgt_frame_face": tensor_frms_face.to(torch.bfloat16), # 主视频 face 帧
            "text": caption,
            "fps": fps // self.frame_interval,
            "ref_image": tensor_ref.to(torch.bfloat16), # ref_image：参考图像
            # "ref_lat": ref_lat[0].to(torch.bfloat16), # ref_lat：参考 latent
            # 'lat': lat[0].to(torch.bfloat16), # lat：主视频 latent
            "token_num": lat[0].numel() / 64, # token_num：主视频 latent token 数
        }
        
        if self.enable_motion_frames:
            item['motion_frame'] = motion_frms.to(torch.bfloat16) # motion_frms：motion 视频帧
            item['tgt_frame_pose'] = tensor_frms_pose.to(torch.bfloat16) # tensor_frms_pose：主视频 pose 视频帧
            item['tgt_frame_face'] = tensor_frms_face.to(torch.bfloat16) # tensor_frms_face：主视频 face 视频帧
            
        if captionemb is not None:
            item['captionemb'] = captionemb # caption embedding
            
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
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]
    
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
        item = self.wrapped_data[new_index]
        if item is not None:
            self.data_buffer[index] = item
        return item

    def __getitem__(self, index):
        if index not in self.data_buffer.keys():
            self.getitem_fast(index)
        return self.data_buffer.pop(index)
    
# 自定义 Sampler，用“token bucket”策略：根据 latent token 数（"token_num"）来选样本，使不同 GPU 上选到的样本 token 数尽量相近。
class TokenBucketSampler(Sampler):
    def __init__(self, dataset, batch_size=1, preload_size=128, world_size=1, rank=0, device="cuda"):
        self.dataset = dataset
        self.preload_size = preload_size
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.batch_size = batch_size

        self.buffer_indices = []  # 保存 index
        self.buffer_tokens = []   # 保存 token 数

        self.cursor = 0
        self.preload() # 先同步预加载一小部分
        self.lock = threading.Lock()

        # 启动 preload 后台线程 持续往缓冲区塞数据
        self._stop_flag = False
        self.thread = threading.Thread(target=self._preload_thread)
        self.thread.daemon = True
        self.thread.start()

    # 在主线程里预加载：不断调用 dataset.getitem_fast(idx)，如果样本非 None，就记录 index 和 token 数。
    def preload(self):
        while len(self.buffer_indices) < self.preload_size // 5:
            idx = self.cursor % len(self.dataset) + len(self.dataset) // self.world_size * self.rank
            sample = self.dataset.getitem_fast(idx)  # ⚠️ 实际读取数据
            if sample is not None:
                token = sample["token_num"]
                self.buffer_indices.append(idx)
                self.buffer_tokens.append(token)
            self.cursor += 1

    # 后台线程持续补充 buffer，始终维护 buffer 大小不超过 preload_size, 用锁保护共享列表
    def _preload_thread(self):
        while not self._stop_flag:
            with self.lock:
                if len(self.buffer_indices) >= self.preload_size:
                    pass  # buffer 满了
                else:
                    idx = self.cursor % len(self.dataset) + len(self.dataset) // self.world_size * self.rank
                    sample = self.dataset.getitem_fast(idx)
                    if sample is not None:
                        token = sample["token_num"]
                        self.buffer_indices.append(idx)
                        self.buffer_tokens.append(token)
                    self.cursor += 1
            time.sleep(0.01)  # 避免空转 CPU 打满

    def __iter__(self):
        
        # 在生成 batch 之前，确保 buffer 至少有一定样本。
        while True:
            while True:
                with self.lock:
                    if len(self.buffer_indices) > 1:
                        break
                time.sleep(0.01)


            # 在分布式训练中：
            # rank0 从 buffer 的 token 数中随机选一个 target_token。
            # 用 dist.broadcast 广播给所有 GPU，让大家都以同一目标 token 为标准。
            if self.rank == 0:
                target_token = random.choice(self.buffer_tokens)
                token_tensor = torch.tensor([target_token], device=self.device, dtype=torch.long)
            else:
                token_tensor = torch.tensor([0], device=self.device, dtype=torch.long)

            dist.broadcast(token_tensor, src=0)
            target_token = token_tensor.item()

            # 每个卡找最接近的 token 样本
            with self.lock:
                diffs = [abs(t - target_token) for t in self.buffer_tokens]
                min_idx = diffs.index(min(diffs))

                chosen_idx = self.buffer_indices[min_idx]
                # ⚠️ 清除选中的样本，避免重复
                del self.buffer_indices[min_idx]
                del self.buffer_tokens[min_idx]
            yield [chosen_idx]
            

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


    def visualize_sample(sample: dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        print("== Visualizing one sample ==")
        print("tgt_frame shape:", sample["tgt_frame"].shape)       # [T, C, H, W]
        print("ref_image shape:", sample["ref_image"].shape)       # [1, C, H, W]
        if "target_frms_face" in sample:
            print("target_frms_face shape:", sample["tgt_frms_face"].shape)

        # 参考帧
        ref_img = tensor_img_to_uint8(sample["ref_image"][0])
        Image.fromarray(ref_img).save(os.path.join(output_dir, "ref_image.png"))

        print(sample.keys())
        # 前 5 帧视频/pose/face可视化
        tgt_frames = sample["tgt_frame"]
        for i in range(min(5, tgt_frames.shape[0])):
            frame = tensor_img_to_uint8(tgt_frames[i])
            Image.fromarray(frame).save(os.path.join(output_dir, f"tgt_frame_{i:03d}.png"))

        face_frames = sample["tgt_frame_face"]
        for i in range(min(5, face_frames.shape[0])):
            face_img = tensor_img_to_uint8(face_frames[i])
            Image.fromarray(face_img).save(os.path.join(output_dir, f"face_target_{i:03d}.png"))

        pose_frames = sample["tgt_frame_pose"]
        for i in range(min(5, pose_frames.shape[0])):
            pose_img = tensor_img_to_uint8(pose_frames[i])
            Image.fromarray(pose_img).save(os.path.join(output_dir, f"tgt_frame_{i:03d}.png"))


        print(f"Saved visualizations to: {output_dir}")
            
    # 示例用法
    # ==== 这里开始真正测试 ====
    dataset = AnimateVideoDataset(
        data_meta_path='/primus_xpfs_workspace_T04/huangyubo/datasets/AVSpeech_part',
        max_num_frames=41,
        meta_name='v1_3_s2v_720_fine',
        dataset_max_token=80000,
        max_token=60000,
        max_wh=720,
        frame_interval=1,
        caption_mode='tarsier',
        force_captionemb=True,
        enable_face_augment=True,
        enable_motion_frames=False,
    )

    print(f"Dataset length: {len(dataset)}")

    save_dir = "/primus_xpfs_workspace_T04/zyx/projects/project2/LiveAnimate_stage1/causvid/datasets/tmp_vis"
    os.makedirs(save_dir, exist_ok=True)
    # ✅ 直接用 dataset[i]，一定会走 __getitem__
    for idx in range(len(dataset)):
        print(f"trying index {idx}")
        sample = dataset[idx]          # 这里就是在调用 __getitem__
        if sample is None:
            print(f"Sample {idx} is None, skip.")
            continue
        visualize_sample(sample, save_dir)
        
        # ======== 维度检查部分开始 ========
        tgt = sample["tgt_frame"]          # [T, C, H, W]
        tgt_pose = sample["tgt_frame_pose"]# [T, C, H, W]
        lat = sample["lat"]                # [C, F_lat, H_lat, W_lat]

        T_img = tgt.shape[0]
        T_pose = tgt_pose.shape[0]
        C_lat, F_lat, H_lat, W_lat = lat.shape

        H_img, W_img = tgt.shape[-2], tgt.shape[-1]

        print("=== shape info ===")
        print(f"tgt_frame      : {tgt.shape}")       # [T, C, H, W]
        print(f"tgt_frame_pose : {tgt_pose.shape}")  # [T, C, H, W]
        print(f"lat (C,F,H,W)  : {lat.shape}")

        # 时间维：按当前写法，理论关系是 T = 4*(F_lat - 1) + 1
        expected_T_from_lat = 4 * (F_lat - 1) + 1
        print(f"T_img={T_img}, T_pose={T_pose}, "
              f"expected_T_from_lat=4*(F_lat-1)+1={expected_T_from_lat}")

        if T_img != T_pose:
            print("[WARN] tgt_frame 和 tgt_frame_pose 的时间长度不一致！")
        if T_img != expected_T_from_lat:
            print("[WARN] 图像帧数和 latent 时间维不满足 4*(F_lat-1)+1 的关系！")

        # 空间维：图像分辨率 vs latent 分辨率比例
        print(f"H_img={H_img}, W_img={W_img}, H_lat={H_lat}, W_lat={W_lat}")
        print(f"H_ratio=H_img/H_lat={H_img / H_lat:.2f}, "
              f"W_ratio=W_img/W_lat={W_img / W_lat:.2f}")
        # 理论上这里接近 8（lat_mul_size）或 16，看预处理配置
        # ======== 维度检查部分结束 ========

        # 如需继续保存可视化可以保留这行
        # visualize_sample(sample, save_dir)

        # 做完一次检查就退出
        # break
    else:
        print("No valid sample found.")