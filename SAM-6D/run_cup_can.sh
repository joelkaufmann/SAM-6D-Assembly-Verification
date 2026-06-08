#!/bin/bash
#SBATCH --job-name=sam6d_cup
#SBATCH --account=3dv
#SBATCH --gpus=2080ti:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=sam6d_cup_%j.log
#SBATCH --error=sam6d_cup_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam6d
module load cuda/12.1.0  

export HYDRA_FULL_ERROR=1

export BASE_DIR="$HOME/SAM-6D/SAM-6D"
# Target the Red Mug (Object 14)
export CAD_PATH="$BASE_DIR/Data/BOP/ycbv/models/obj_000014.ply" 
export RGB_PATH="$BASE_DIR/Data/BOP/ycbv/test/000048/rgb/000001.png"
export DEPTH_PATH="$BASE_DIR/Data/BOP/ycbv/test/000048/depth/000001.png"
# Save to a new folder
export OUTPUT_DIR="$BASE_DIR/Data/BOP/ycbv/test/000048/outputs_obj14"

export BOP_CAM_PATH="$BASE_DIR/Data/BOP/ycbv/test/000048/scene_camera.json"
export CAMERA_PATH="$BASE_DIR/Data/BOP/ycbv/test/000048/camera_custom.json"

# Extract camera params
python -c "import json; d=json.load(open('$BOP_CAM_PATH')); json.dump(list(d.values())[0], open('$CAMERA_PATH', 'w'))"

echo "Starting Pipeline for Object 14 (Mug)..."

# 1. Render Templates (Required for the new object)
cd $BASE_DIR/Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH 

# 2. Instance Segmentation
cd $BASE_DIR/Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model sam --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH

# 3. Pose Estimation
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json
cd $BASE_DIR/Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

echo "Done! Check outputs_obj14 folder."