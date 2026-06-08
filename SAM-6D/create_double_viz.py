import argparse
import json
import numpy as np
import cv2
import os
import sys
import trimesh
import pycocotools.mask as cocomask

# --- 1. Import SAM6D Native Drawing Tools ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, 'Pose_Estimation_Model')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from draw_utils import draw_detections

# --- 2. Setup Command Line Arguments ---
def get_args():
    parser = argparse.ArgumentParser(description="Generate Assembly Verification Visualizations")
    parser.add_argument("--masks", action="store_true", help="Overlay 2D segmentation masks")
    parser.add_argument("--boxes", action="store_true", help="Draw 3D bounding boxes")
    parser.add_argument("--axes", action="store_true", help="Draw 3D coordinate frames (XYZ arrows)")
    parser.add_argument("--output", type=str, default="midterm_presentation_image.png", help="Name of the output file")
    return parser.parse_args()

# --- 3. Drawing Helper Functions ---
def overlay_mask(image_bgr, json_path, color_bgr, alpha=0.5):
    with open(json_path, 'r') as f:
        dets = json.load(f)
    best_det = max(dets, key=lambda x: x['score'])
    seg = best_det['segmentation']
    
    h, w = seg['size']
    try:
        rle = cocomask.frPyObjects(seg, h, w)
    except:
        rle = seg
    mask = cocomask.decode(rle) 
    
    for c in range(3):
        image_bgr[:, :, c] = np.where(mask == 1,
                                      image_bgr[:, :, c] * (1 - alpha) + alpha * color_bgr[c],
                                      image_bgr[:, :, c])
    return image_bgr

def draw_coordinate_frame(img_bgr, R, t, K, length_mm=60.0):
    points_3d = np.array([
        [0.0, 0.0, 0.0],          # Origin
        [length_mm, 0.0, 0.0],    # X-axis (Red)
        [0.0, length_mm, 0.0],    # Y-axis (Green)
        [0.0, 0.0, length_mm]     # Z-axis (Blue)
    ])

    R_matrix = np.array(R).reshape(3, 3)
    t_vector = np.array(t).reshape(3, 1)
    points_3d_transformed = (R_matrix @ points_3d.T) + t_vector

    points_2d = K @ points_3d_transformed
    points_2d = (points_2d[:2, :] / points_2d[2, :]).T.astype(int)

    origin = tuple(points_2d[0])
    cv2.line(img_bgr, origin, tuple(points_2d[1]), (0, 0, 255), 3) # X - Red (BGR)
    cv2.line(img_bgr, origin, tuple(points_2d[2]), (0, 255, 0), 3) # Y - Green (BGR)
    cv2.line(img_bgr, origin, tuple(points_2d[3]), (255, 0, 0), 3) # Z - Blue (BGR)
    return img_bgr

# --- 4. Main Execution ---
def main():
    args = get_args()
    
    # Check if user selected at least one option
    if not (args.masks or args.boxes or args.axes):
        print("Please specify what to draw! Use --masks, --boxes, or --axes (or a combination).")
        return

    # Hardcoded paths for Scene 48
    base_path = "Data/BOP/ycbv/test/000048/"
    rgb_path = base_path + "rgb/000001.png"
    cam_path = base_path + "camera_custom.json"

    can_json = base_path + "outputs/sam6d_results/detection_pem.json"
    can_cad  = "Data/BOP/ycbv/models/obj_000001.ply"
    
    cup_json = base_path + "outputs_obj14/sam6d_results/detection_pem.json"
    cup_cad  = "Data/BOP/ycbv/models/obj_000014.ply"

    # Load base image and camera data
    img_bgr = cv2.imread(rgb_path)
    cam_info = json.load(open(cam_path))
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    # --- Step A: Masks (Uses BGR) ---
    if args.masks:
        print("-> Painting segmentation masks...")
        img_bgr = overlay_mask(img_bgr, can_json, color_bgr=(0, 0, 255)) # Red
        img_bgr = overlay_mask(img_bgr, cup_json, color_bgr=(0, 255, 0)) # Green

    # --- Step B: Coordinate Axes (Uses BGR) ---
    if args.axes:
        print("-> Drawing 3D coordinate axes...")
        with open(can_json) as f: can_data = json.load(f)[0]
        with open(cup_json) as f: cup_data = json.load(f)[0]
        img_bgr = draw_coordinate_frame(img_bgr, can_data['R'], can_data['t'], K)
        img_bgr = draw_coordinate_frame(img_bgr, cup_data['R'], cup_data['t'], K)

    # --- Step C: Bounding Boxes (Requires RGB conversion for SAM6D's tool) ---
    if args.boxes:
        print("-> Drawing 3D bounding boxes...")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        def overlay_pose(img_array, json_file, cad_file, color_rgb):
            with open(json_file) as f: data = json.load(f)[0]
            pred_rot = np.array([data['R']])
            pred_trans = np.array([data['t']])
            mesh = trimesh.load_mesh(cad_file)
            model_points = mesh.sample(2048).astype(np.float32)
            return draw_detections(img_array, pred_rot, pred_trans, model_points, np.array([K]), color=color_rgb)

        img_rgb = overlay_pose(img_rgb, can_json, can_cad, color_rgb=(255, 0, 0)) # Red
        img_rgb = overlay_pose(img_rgb, cup_json, cup_cad, color_rgb=(0, 255, 0)) # Green
        
        # Convert back to BGR so OpenCV can save it correctly
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Save output
    cv2.imwrite(args.output, img_bgr)
    print(f"\nSuccess! Visual saved to: {args.output}")

if __name__ == '__main__':
    main()