#!/bin/bash
#SBATCH --job-name=sam6d_ycbv
#SBATCH --account=3dv
#SBATCH --gpus=2080ti:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=sam6d_ycbv_%j.log
#SBATCH --error=sam6d_ycbv_%j.err

# 1. Initialize environment and load CUDA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam6d
module load cuda/12.1.0  

export HYDRA_FULL_ERROR=1

# 2. Set the exact paths for YCB-V Scene 000048
export BASE_DIR="$HOME/SAM-6D/SAM-6D"
export CAD_PATH="$BASE_DIR/Data/BOP/ycbv/models/obj_000001.ply" 
export RGB_PATH="$BASE_DIR/Data/BOP/ycbv/test/000048/rgb/000001.png"
export DEPTH_PATH="$BASE_DIR/Data/BOP/ycbv/test/000048/depth/000001.png"
export OUTPUT_DIR="$BASE_DIR/Data/BOP/ycbv/test/000048/outputs"

# --- THE FIX: Create a simplified camera.json for the custom script ---
export BOP_CAM_PATH="$BASE_DIR/Data/BOP/ycbv/test/000048/scene_camera.json"
export CAMERA_PATH="$BASE_DIR/Data/BOP/ycbv/test/000048/camera_custom.json"

# This Python one-liner extracts the nested camera info and saves it in the required format
python -c "import json; d=json.load(open('$BOP_CAM_PATH')); json.dump(list(d.values())[0], open('$CAMERA_PATH', 'w'))"
# ----------------------------------------------------------------------

# 3. Run the custom inference demo
cd $BASE_DIR
echo "Starting SAM6D on 2080ti GPU for Object 000001..."
sh demo.sh