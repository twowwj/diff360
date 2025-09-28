#!/bin/sh
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH -N 1
#SBATCH -t 0-01:00

#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err

# source /home/weijie.wang/.conda/envs/diffportrait360/bin/activate

source /home/${USER}/.bashrc
conda activate diff360

# Set CUDA environment after conda activation
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Change directory and set GPU
cd /home/weijie.wang/project/DiffPortrait360/diffportrait360_release/code
export CUDA_VISIBLE_DEVICES=0
# conda activate /home/weijie.wang/.conda/envs/diffportrait360



# Step 0.0 :Put the you own image to test under folder sample_data/input_image
echo "Start Inference"

# Step 0.1: Using 3DDFA_V2_cropping to get camera pose and crop the image to the right format

model="easy-khair-180-gpc0.8-trans10-025000.pkl"
out="../../sample_data/3DNoise"
target_img="../../sample_data/input_image"

# change the path to your own
PANO_HEAD_MODEL="/home/weijie.wang/project/DiffPortrait360/diffportrait360/easy-khair-180-gpc0.8-trans10-025000.pkl"
Head_Back_MODEL="/home/weijie.wang/project/DiffPortrait360/diffportrait360/back_head-230000.th"
Diff360_MODEL="/home/weijie.wang/project/DiffPortrait360/diffportrait360/model_state-340000.th"

# Step1: PanoHead 3D aware noise generation
#cd to 3DNOise Generation folder in order to get the 3D aware noise from PanoHead PTI
cd 3DNoise
python projector_withseg.py \
--outdir=${out} \
--num_steps 200 \
--target_img=${target_img} \
--network ${model} \
--camera_json ../../sample_data/input_image/dataset.json \
--network ${PANO_HEAD_MODEL}

cd ..

# Step2: Generate Head_Back with attention visualization
torchrun --master_port 14020 inference.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml \
--test_dataset back_head_generation \
--control_mode controlnet_important \
--local_image_dir ../sample_data/Back_Head \
--resume_dir ${Head_Back_MODEL} \
--control_type GAN_Generated \
--inference_image_path ../sample_data/input_image \
--nSample 1 \
--condition_path ../sample_data/cam_condition/sphere32 \



# # check sample_data/Back_Head folder to see if the result is correct
# # # Step3: Generate Video

torchrun --master_port 14031 inference.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml \
--test_dataset full_head_clean_inference_final_face \
--control_mode controlnet_important \
--local_image_dir ../sample_data/result \
--resume_dir ${Diff360_MODEL} \
--control_type GAN_Generated \
--inference_image_path ../sample_data \
--nSample 8 \
--condition_path ../sample_data/cam_condition/sphere32 \
--denoise_from_guidance \
--initial_image_path ../sample_data/3DNoise \

$@

