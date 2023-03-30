python ../train.py \
   --mode train \
   --dataset_name klevr \
   --root_dir /home/timothy/Desktop/2023Spring/my_nerf/nesf_dataset/klevr/0 \
   --N_importance 64 --img_wh 256 256 --noise_std 0 \
   --num_epochs 16 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --exp_name exp_nesf