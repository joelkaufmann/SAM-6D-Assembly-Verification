#!/bin/bash
#SBATCH --job-name=sam6d_pipe
#SBATCH --account=3dv
#SBATCH --gpus=2080ti:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00 
#SBATCH --output=sam6d_pipe_%j.log
#SBATCH --error=sam6d_pipe_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam6d
module load cuda/12.1.0  

export HYDRA_FULL_ERROR=1
export BASE_DIR="$HOME/SAM-6D/SAM-6D"

# --- COMMON DATASET PATHS ---
export DATASET_DIR="$BASE_DIR/Data/BOP/pipeconnector"
export SCENE_DIR="$DATASET_DIR/test/000024"

# We use the 4th frame of your scene for the inference
export RGB_PATH="$SCENE_DIR/rgb/000013.png"
export DEPTH_PATH="$SCENE_DIR/depth/000013.png"

export BOP_CAM_PATH="$SCENE_DIR/scene_camera.json"
export CAMERA_PATH="$SCENE_DIR/camera_custom.json"

# Extract camera params into SAM-6D's expected format
python -c "import json; d=json.load(open('$BOP_CAM_PATH')); json.dump(list(d.values())[0], open('$CAMERA_PATH', 'w'))"

# ==========================================
# PART 1: The Base Connector (Object 1)
# ==========================================
echo "Starting Pipeline for Part 1 (Base Connector)..."
export CAD_PATH_1="$DATASET_DIR/models/obj_000001.ply" 
export OUTPUT_DIR_1="$SCENE_DIR/outputs_part1"

# 1. Render Templates
cd $BASE_DIR/Render
# blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR_1 --cad_path $CAD_PATH_1 

# 2. Instance Segmentation (ISM)
cd $BASE_DIR/Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model sam --output_dir $OUTPUT_DIR_1 --cad_path $CAD_PATH_1 --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH

# INJECT POST-PROCESSING HERE
export SEG_PATH_1=$OUTPUT_DIR_1/sam6d_results/detection_ism.json
cd $BASE_DIR
python postprocess_ism.py --json_path $SEG_PATH_1 --rgb_path $RGB_PATH --target nut

# 3. Pose Estimation (PEM)
export SEG_PATH_1=$OUTPUT_DIR_1/sam6d_results/detection_ism.json
cd $BASE_DIR/Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR_1 --cad_path $CAD_PATH_1 --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH_1

# ==========================================
# PART 2: The Insert (Object 2)
# ==========================================
echo "Starting Pipeline for Part 2 (Insert)..."
export CAD_PATH_2="$DATASET_DIR/models/obj_000002.ply" 
export OUTPUT_DIR_2="$SCENE_DIR/outputs_part2"

# 1. Render Templates
cd $BASE_DIR/Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR_2 --cad_path $CAD_PATH_2 

# 2. Instance Segmentation (ISM)
cd $BASE_DIR/Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model sam --output_dir $OUTPUT_DIR_2 --cad_path $CAD_PATH_2 --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH

# INJECT POST-PROCESSING HERE
export SEG_PATH_1=$OUTPUT_DIR_1/sam6d_results/detection_ism.json
cd $BASE_DIR
python postprocess_ism.py --json_path $SEG_PATH_1 --rgb_path $RGB_PATH --target pipe

# 3. Pose Estimation (PEM)
export SEG_PATH_2=$OUTPUT_DIR_2/sam6d_results/detection_ism.json
cd $BASE_DIR/Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR_2 --cad_path $CAD_PATH_2 --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH_2

# ==========================================
# PART 3: Visualization & Metrics
# ==========================================
echo "Running Visualization and Metrics..."
cd $BASE_DIR
# Notice how we now pass the scene and frame directly into the python script!
python create_viz_pipeconnector.py --scene 000024 --frame 000013 --boxes --masks --axes --output pipe_assembly_scene24_frame130.png

echo "All Done! Check pipe_assembly_scene24_frame130.png."