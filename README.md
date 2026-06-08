# <p align="center"> CAD-Conditioned Egocentric 3D Reconstruction for Assembly Verification </p>

#### <p align="center"> Joel Kaufmann, Timon Mettler </p>
#### <p align="center"> ETH Zurich — 3D Vision Project </p>

This repository extends the **SAM-6D** framework (CVPR 2024) to handle egocentric 3D assembly verification using Project Aria data. It introduces specialized post-processing steps to overcome zero-shot segmentation boundaries and overlays perspective-correct 3D dimensioning to measure precise millimeter-level insertion depths.

<p align="center">
  <img width="50%" src="SAM-6D/pipe_assembly_viz.png" alt="3D Perspective Coaxial Assembly Verification Overlay"/>
</p>

## Overview
In high-precision assembly verification (such as solar connectors or industrial piping), standard zero-shot frameworks face two critical barriers:
1. **Mask Merging:** Visually ambiguous sub-components (e.g., a base pipe and its locking nut coated in protective/scanning spray) are frequently merged into a single "fried egg" instance mask by foundational models.
2. **Perspective Distortion:** Evaluating 3D displacement parameters directly from flat 2D pixel layouts distorts real-world physical dimensions.

This extension introduces an intervened, end-to-end pipeline:
- **Foundational Detection:** Utilizes SAM-6D's Instance Segmentation Model (ISM) to generate initial structural object proposals.
- **Morphological Color-Splitting Intermediary (`postprocessing.py`):** Automatically intercepts merged foundational masks, applies custom dual-band HSV color slicing, performs morphological opening to clean scanning halos, and applies a spatial dilation buffer to cleanly isolate independent sub-components.
- **Pose Estimation Model (PEM):** Computes robust 3D translation ($t$) and rotation ($R$) parameters for the split masks using target `.ply` CAD geometry.
- **Perspective-Correct Dimensioning Visualizer (`create_viz_pipeconnector.py`):** Maps 3D component transformations back to the 2D plane, rendering spatial leader lines and a CAD-style dimension layout aligned strictly with the objects' coaxial local Z-axis.

---

## Getting Started

### 1. Environment Setup
Clone this repository and restore the configured environment matrix:
```bash
git clone [https://github.com/joelkaufmann/SAM-6D-Assembly-Verification.git](https://github.com/joelkaufmann/SAM-6D-Assembly-Verification.git)
cd SAM-6D-Assembly-Verification
conda env create -f SAM-6D/environment.yaml
conda activate sam6d
```

### 2. Running the Core Pipeline
Execute the foundational estimation step on your target scene and frame sequence:
```bash
bash SAM-6D/run_pipeconnector.sh
```

### 3. Executing the Morphological Mask Splitter
If the instance segmenter yields a single merged mask across the assembly interface, pass the generated JSON file to the color post-processor to isolate the individual target items (e.g., `nut` or `pipe` targeting category IDs):
```bash
python SAM-6D/postprocessing.py \
  --json_path Data/BOP/pipeconnector/test/000024/outputs_part1/sam6d_results/detection_ism.json \
  --rgb_path Data/BOP/pipeconnector/test/000024/rgb/000015.png \
  --target pipe
```
*Note: This script safely archives a pristine copy of your original inference arrays as `_RAW.json`, allowing you to run, tweak parameters, and evaluate multiple times without re-running the heavy neural network inference layer.*

### 4. Generating the 3D Verification Visuals
Compute the perspective-correct relative transformation matrix ($T_{rel} = T_{base}^{-1} \times T_{insert}$) and overlay the CAD-style 3D dimension tracking lines:
```bash
python SAM-6D/create_viz_pipeconnector.py \
  --scene 000024 \
  --frame 000015 \
  --boxes \
  --masks \
  --axes \
  --output pipe_assembly_scene24_frame15.png
```

---

## Workspace Directory Structure
The repository is structured to separate raw platform build processes from core engineering modules:
```text
SAM-6D-Assembly-Verification/
│
├── SAM-6D/
│   ├── Instance_Segmentation_Model/    # ISM baseline architecture
│   ├── Pose_Estimation_Model/          # PEM pointcloud alignment networks
│   │
│   ├── postprocessing.py               # Custom HSV & morphological mask extraction layer
│   ├── create_viz_pipeconnector.py     # Coaxial 3D dimension calculator & visualizer 
│   ├── run_pipeconnector.sh            # Custom runner shell automation script
│   └── environment.yaml                # Verified dependency matrix config
│
└── .gitignore                          # Configured to automatically isolate heavy datasets/weights
```

---

## Acknowledgments & Base Citation
This project builds heavily upon the core open-vocabulary framework established by **SAM-6D**:

```bibtex
@article{lin2023sam,
  title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
  author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
  journal={arXiv preprint arXiv:2311.15707},
  year={2023}
}
```
