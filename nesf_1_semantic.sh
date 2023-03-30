CUDA_LAUNCH_BLOCKING=1 python -m pdb train_semantic.py --dataset_name klevr \
    --root_dir /home/timothy/Desktop/2023Spring/GeoNeRF/data/data/nesf_data/klevr/1 \
    --N_importance 64 --img_wh 256 256 --noise_std 0 \
    --num_epochs 16 --batch_size 1024 \
    --optimizer adam --lr 5e-4 \
    --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
    --exp_name exp_nesf_semantic \
    --loss_type crossentropy --semantic_class 6 --resolution 128 \
    --nerf_ckpt /home/timothy/Desktop/2023Spring/my_nerf/ckpts/nesf1_calculate_near_far_revise/epoch=15-step=215040.ckpt \
    