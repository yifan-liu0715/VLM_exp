# 我使用下面的脚本进行训练
cd /nvmedata/workspace2/users/nina/prj/DiffSynth-Studio

conda activate v2v_train
export PYTHONPATH=$(pwd):$PYTHONPATH
export DIFFSYNTH_SKIP_DOWNLOAD=true

CUDA_VISIBLE_DEVICES=1 python examples/wanvideo/model_training/train.py \
    --task "sft" \
    --dataset_base_path ./data/dataset_video_train \
    --dataset_metadata_path ./data/dataset_video_train/metadata_train.csv \
    --data_file_keys "video" \
    --model_paths '[
        "./pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        [
            "./pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "./pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "./pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
        ],
        "./pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    ]' \
    --tokenizer_path ./pretrain_weights/Wan2.2-TI2V-5B/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl/ \
    --extra_inputs "input_image" \
    --trainable_models dit \
    --lora_base_model dit \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --output_path ./output_models/wan2.2_lora_test \
    --num_epochs 3 \
    --gradient_accumulation_steps 4 \
    --use_gradient_checkpointing \
    --learning_rate 1e-4