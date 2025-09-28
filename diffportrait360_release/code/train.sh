#!/bin/bash

#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH -N 1
#SBATCH -t 24:00:00

#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err



python -m torch.distributed.run --master_port 12355 --master_addr 'localhost' --nproc_per_node 1 train.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml \
--output_dir  ./control \
--train_batch_size 1 \
--num_workers 4 \
--img_bin_limit 10 \
--control_mode controlnet_important \
--control_type GAN_Generated \
--train_dataset full_head_clean_real_data_single_frame \
--with_text \
--local_image_dir ./image_log \
--local_log_dir ./log_board \
--finetune_attn \
--finetune_control \
--save_steps 500000 \
--resume_dir /home/weijie.wang/project/DiffPortrait360/diffportrait360/back_head-230000.th \
--dataset_root_path /home/weijie.wang/project/Dataset/data/PanoHead_out \
$@
# change to your ckpt path and your datapath


# train mm


