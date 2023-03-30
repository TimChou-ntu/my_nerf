python ../train.py \
   --mode train \
   --dataset_name klevr \
   --root_dir /home/timothy/Desktop/2023Spring/GeoNeRF/data/data/nesf_data/klevr/1 \
   --N_importance 64 --img_wh 256 256 --noise_std 0 \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --exp_name nesf1_calculate_near_far_revise_longer \
   # --nerf_ckpt /home/timothy/Desktop/2023Spring/my_nerf/ckpts/nesf1_white_back_false/epoch=14-step=201600.ckpt
   # line 9 is for mode test, will create validation image on ./results/ folder \
