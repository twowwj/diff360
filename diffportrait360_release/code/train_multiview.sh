#!/bin/bash

#SBATCH -p long
#SBATCH --nodelist=gpu01  
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8192
#SBATCH -N 1
#SBATCH -t 24:00:00

#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err



python -m torch.distributed.run --master_port 12360 --master_addr 'localhost' --nproc_per_node 1 train_multiview_consistency.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml \
--image_pretrain_dir /home/weijie.wang/project/DiffPortrait360/diffportrait360/weight/control_v11f1e_sd15_tile.pth \
--pose_pretrain_dir  /home/weijie.wang/project/DiffPortrait360/diffportrait360/weight/control_v11f1p_sd15_depth.pth \
--dataset_root_path /home/weijie.wang/project/Dataset/data/PanoHead_out \
--train_dataset full_head_clean_real_data_temporal \
--train_batch_size 1 \
--num_workers 2 \
--img_bin_limit 10 \
--control_mode controlnet_important \
--control_type GAN_Generated \
--with_text \
--wonoise \
--mask_bg \
--more_image_control \
--finetune_imagecond_unet \
--save_steps 500 \
--logging_gen_steps 500 \
--local_image_dir ./consistency_image_log \
--local_log_dir ./consistency_log \
--output_dir  ./consistency_output \
$@

# change to your ckpt path and your datapath


# train mm
