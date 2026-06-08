#!/bin/bash
#SBATCH --job-name=render_tless
#SBATCH --account=3dv
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=render_output.log
#SBATCH --error=render_error.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam6d

# Load CUDA
module load cuda/12.1.0  

# Force headless GPU rendering
export PYOPENGL_PLATFORM=egl

cd /home/jkaufmann/SAM-6D/SAM-6D/Render

echo "Starting GPU Render..."
blenderproc run render_bop_templates.py --dataset_name tless

# Auto-rename the folder so the inference script can find it!
echo "Rendering complete! Renaming folder for SAM-6D..."
mv /home/jkaufmann/SAM-6D/SAM-6D/Data/BOP/BOP-Templates /home/jkaufmann/SAM-6D/SAM-6D/Data/BOP/templates_pyrender

echo "All done! Ready for inference."