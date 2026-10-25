import torch
from PIL import Image
from diffsynth.core import load_state_dict
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video_causal_animate_training_motion import WanVideoPipeline, ModelConfig
import torch
import numpy as np
from PIL import Image
import os

from lora_helper import add_lora_to_model, fuse_lora_to_base_model
from diffsynth.core import load_state_dict

torch.cuda.set_device(0)
# Load models
model_paths = [["/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00001-of-00004.safetensors", "/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00002-of-00004.safetensors", "/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00003-of-00004.safetensors", "/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00004-of-00004.safetensors"], "/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B/models_t5_umt5-xxl-enc-bf16.pth", "/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B/Wan2.1_VAE.pth", "/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]
tokenizer_path = "/primus_xpfs_workspace_T08/zhangyuxuan/models/Wan2.2-Animate-14B/google/umt5-xxl"
default_dtype = torch.bfloat16

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_paths=model_paths,
    tokenizer_path=tokenizer_path,
)
pipe.to("cuda")
pipe.dit.local_attn_size = 9

# add lora only 
pipe.dit = add_lora_to_model(
    pipe.dit,
    lora_rank=128,
    lora_target_modules="q,k,v,o,ffn.0,ffn.2",
    pretrained_lora_path="/primus_xpfs_workspace_T08/zhangyuxuan/projects/LiveAnimate_stage1_0401/output_fsdplora_attn9_tsf_2500samples_lr5e-5/step-5000.safetensors",
)

# input_image = Image.open("/primus_xpfs_workspace_T06/zhangyuxuan/projects1/LA_inference_motion/process_results/src_ref.png").resize((720, 400))
# animate_pose_video = VideoData("/primus_xpfs_workspace_T06/zhangyuxuan/projects1/LA_inference_motion/process_results/src_pose.mp4").raw_data()[:49]
# animate_pose_video = [frame.resize((720, 400)) for frame in animate_pose_video]
# animate_face_video = VideoData("/primus_xpfs_workspace_T06/zhangyuxuan/projects1/LA_inference_motion/process_results/src_face.mp4").raw_data()[:49]
# animate_face_video = [frame.resize((512, 512)) for frame in animate_face_video]  # ✅ 补上 resize


# input_image = Image.open("/primus_xpfs_workspace_T08/zhangyuxuan/projects1/000006/src_ref.png").resize((512, 800))
# animate_pose_video = VideoData("/primus_xpfs_workspace_T08/zhangyuxuan/projects1/000006/src_pose.mp4").raw_data()[:76]
# animate_pose_video = [frame.resize((512, 800)) for frame in animate_pose_video]
# animate_face_video = VideoData("/primus_xpfs_workspace_T08/zhangyuxuan/projects1/000006/src_face.mp4").raw_data()[:76]
# animate_face_video = [frame.resize((512,512)) for frame in animate_face_video]  # ✅ 补上 resize


input_image = Image.open("/primus_xpfs_workspace_T08/zhangyuxuan/projects1/LA_stage2_0228/minimal_inference/process_results1/src_ref.png").resize((512, 800))
animate_pose_video = VideoData("/primus_xpfs_workspace_T08/zhangyuxuan/projects1/LA_stage2_0228/minimal_inference/process_results1/src_pose.mp4").raw_data()[:76]
animate_pose_video = [frame.resize((512, 800)) for frame in animate_pose_video]
animate_face_video = VideoData("/primus_xpfs_workspace_T08/zhangyuxuan/projects1/LA_stage2_0228/minimal_inference/process_results1/src_face.mp4").raw_data()[:76]
animate_face_video = [frame.resize((512,512)) for frame in animate_face_video]  # ✅ 补上 resize

motion_size = 0
chunk_size = 76

def chunk_with_padding(frames, chunk_size=12):
    chunks = []
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + [chunk[-1]] * (chunk_size - len(chunk))
        chunks.append(chunk)
    return chunks

pose_chunks = chunk_with_padding(animate_pose_video, chunk_size=chunk_size)
face_chunks = chunk_with_padding(animate_face_video, chunk_size=chunk_size)
print(f"总共分成 {len(pose_chunks)} 个chunk，每块 {len(pose_chunks[0])} 帧")

videos = []
motion_video_cur = [input_image] * motion_size  # ✅ 第一次用参考图填充

for chunk_idx, (pose_chunk, face_chunk) in enumerate(zip(pose_chunks, face_chunks)):
    print("chunk_idx:", chunk_idx)
    start_frame = chunk_idx * chunk_size  # ✅ 传入正确的起始帧位置
    video = pipe(
        prompt="视频中的人在做动作",
        seed=0, tiled=True,
        input_image=input_image,
        motion_video=motion_video_cur,   # ✅ 上一个 chunk 的输出帧
        animate_pose_video=pose_chunk,
        animate_face_video=face_chunk,
        num_frames=chunk_size, height=800, width=512,
        # num_frames=chunk_size, height=400, width=720,
        num_inference_steps=20, cfg_scale=5,
        start_frame=start_frame,         # ✅ 修正时序位置编码
    )
    video = video[motion_size:]
    videos.append(video) # 1 sink + 13 motion 
    # ✅ 用当前输出作为下一次的 motion_video（取最后9帧）
    # input_image = video[-motion_size+1:-motion_size]
    motion_video_cur = video[-motion_size:] if len(video) >= motion_size else video + [video[-1]] * (motion_size - len(video))

output_path = "./causal_output_ahead_3/"
os.makedirs(output_path, exist_ok=True)

# ✅ 拼接所有 chunk 保存完整视频
all_frames = []
for v in videos:
    all_frames.extend(v)
save_video(all_frames, output_path + "video_1_Wan2.2-Animate-14B.mp4", fps=16, quality=5)
save_video(animate_pose_video, output_path + "control_1_Wan2.2-Animate-14B.mp4", fps=16, quality=5)
save_video(animate_face_video, output_path + "face_1_Wan2.2-Animate-14B.mp4", fps=16, quality=5)