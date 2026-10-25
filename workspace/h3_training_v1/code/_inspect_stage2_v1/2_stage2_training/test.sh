    python test_teacher_vis.py \
        --output_dir ./test_teacher_vis_out \
        --num_steps 10 \
        --num_samples 3 \
        2>&1 | tee -a ./test_teacher_vis_out/infer.log