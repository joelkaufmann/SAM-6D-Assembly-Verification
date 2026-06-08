# import argparse
# import json
# import numpy as np
# import cv2
# import os
# import sys
# import trimesh
# import pycocotools.mask as cocomask
# from scipy.spatial.transform import Rotation as R

# # --- 1. Import SAM6D Native Drawing Tools ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.join(BASE_DIR, 'Pose_Estimation_Model')
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# from draw_utils import draw_detections

# # --- 2. Setup Command Line Arguments ---
# def get_args():
#     parser = argparse.ArgumentParser(description="Assembly Verification: Metrics & Visualization")
#     parser.add_argument("--scene", type=str, required=True, help="Scene folder name (e.g., 000005)")
#     parser.add_argument("--frame", type=str, required=True, help="Frame ID (e.g., 000010)")
    
#     parser.add_argument("--masks", action="store_true", help="Overlay 2D segmentation masks")
#     parser.add_argument("--boxes", action="store_true", help="Draw 3D bounding boxes")
#     parser.add_argument("--axes", action="store_true", help="Draw 3D coordinate frames")
#     parser.add_argument("--output", type=str, default="pipe_assembly_viz.png", help="Output file name")
#     return parser.parse_args()

# # --- 3. Helper Functions ---
# def make_4x4(R_list, t_list):
#     T = np.eye(4)
#     T[:3, :3] = np.array(R_list).reshape(3, 3)
#     T[:3, 3] = np.array(t_list)
#     return T

# def overlay_mask(image_bgr, json_path, color_bgr, alpha=0.5):
#     with open(json_path, 'r') as f:
#         dets = json.load(f)
#     # Grab the mask with the highest confidence score
#     best_det = max(dets, key=lambda x: x['score'])
#     seg = best_det['segmentation']
    
#     h, w = seg['size']
#     try:
#         rle = cocomask.frPyObjects(seg, h, w)
#     except:
#         rle = seg
#     mask = cocomask.decode(rle) 
    
#     for c in range(3):
#         image_bgr[:, :, c] = np.where(mask == 1,
#                                       image_bgr[:, :, c] * (1 - alpha) + alpha * color_bgr[c],
#                                       image_bgr[:, :, c])
#     return image_bgr

# def draw_coordinate_frame(img_bgr, R_mat, t_vec, K, length_mm=60.0):
#     points_3d = np.array([
#         [0.0, 0.0, 0.0],          
#         [length_mm, 0.0, 0.0],    
#         [0.0, length_mm, 0.0],    
#         [0.0, 0.0, length_mm]     
#     ])

#     R_matrix = np.array(R_mat).reshape(3, 3)
#     t_vector = np.array(t_vec).reshape(3, 1)
#     points_3d_transformed = (R_matrix @ points_3d.T) + t_vector

#     points_2d = K @ points_3d_transformed
#     points_2d = (points_2d[:2, :] / points_2d[2, :]).T.astype(int)

#     origin = tuple(points_2d[0])
#     cv2.line(img_bgr, origin, tuple(points_2d[1]), (0, 0, 255), 3) # X - Red
#     cv2.line(img_bgr, origin, tuple(points_2d[2]), (0, 255, 0), 3) # Y - Green
#     cv2.line(img_bgr, origin, tuple(points_2d[3]), (255, 0, 0), 3) # Z - Blue
#     return img_bgr

# # --- 4. Main Execution ---
# def main():
#     args = get_args()
    
#     if not (args.masks or args.boxes or args.axes):
#         print("Please specify what to draw! Use --masks, --boxes, or --axes.")
#         return

#     # --- DYNAMIC DATASET PATHS ---
#     base_path = f"Data/BOP/pipeconnector/test/{args.scene}/"
#     FRAME_ID = args.frame
    
#     rgb_path = base_path + f"rgb/{FRAME_ID}.png"
#     cam_path = base_path + "scene_camera.json"

#     part1_json = base_path + "outputs_part1/sam6d_results/detection_pem.json"
#     part1_cad  = "Data/BOP/pipeconnector/models/obj_000001.ply"
    
#     part2_json = base_path + "outputs_part2/sam6d_results/detection_pem.json"
#     part2_cad  = "Data/BOP/pipeconnector/models/obj_000002.ply"

#     # Load Image and Camera Intrinsics
#     img_bgr = cv2.imread(rgb_path)
#     if img_bgr is None:
#         print(f"Error: Could not load image at {rgb_path}")
#         return

#     # Dynamically extract camera matrix
#     cam_data_all = json.load(open(cam_path))
#     cam_key = list(cam_data_all.keys())[0] 
#     K = np.array(cam_data_all[cam_key]['cam_K']).reshape(3, 3)

#     # --- FIX APPLIED HERE: Load Poses by highest score ---
#     with open(part1_json) as f: 
#         part1_data = max(json.load(f), key=lambda x: x['score'])
#     with open(part2_json) as f: 
#         part2_data = max(json.load(f), key=lambda x: x['score'])

#     # ==========================================
#     # SECTION A: Calculate Relative Distances
#     # ==========================================
#     T_cam_part1 = make_4x4(part1_data['R'], part1_data['t'])
#     T_cam_part2 = make_4x4(part2_data['R'], part2_data['t'])

#     T_part1_part2 = np.linalg.inv(T_cam_part1) @ T_cam_part2

#     t_rel = T_part1_part2[:3, 3]
#     R_rel_matrix = T_part1_part2[:3, :3]
#     r_euler = R.from_matrix(R_rel_matrix).as_euler('xyz', degrees=True)

#     print(f"\n--- Assembly Verification Metrics (Scene {args.scene}) ---")
#     print("Target: Insert (Part 2) relative to Base (Part 1)\n")
#     print(f"1. Relative Position Offset (mm):")
#     print(f"   X (Left/Right) : {t_rel[0]:.1f} mm")
#     print(f"   Y (Up/Down)    : {t_rel[1]:.1f} mm")
#     print(f"   Z (Front/Back) : {t_rel[2]:.1f} mm")
#     print(f"   Total Distance : {np.linalg.norm(t_rel):.1f} mm\n")
    
#     print(f"2. Relative Rotation/Twist (Degrees):")
#     print(f"   Pitch (X-axis) : {r_euler[0]:.1f} deg")
#     print(f"   Yaw   (Y-axis) : {r_euler[1]:.1f} deg")
#     print(f"   Roll  (Z-axis) : {r_euler[2]:.1f} deg")
#     print("-------------------------------------\n")

#     # ==========================================
#     # SECTION B: Drawing the Visualization
#     # ==========================================
#     if args.masks:
#         print("-> Painting segmentation masks...")
#         img_bgr = overlay_mask(img_bgr, part1_json, color_bgr=(0, 0, 255)) # Red Base
#         img_bgr = overlay_mask(img_bgr, part2_json, color_bgr=(0, 255, 0)) # Green Insert

#     if args.axes:
#         print("-> Drawing 3D coordinate axes...")
#         img_bgr = draw_coordinate_frame(img_bgr, part1_data['R'], part1_data['t'], K)
#         img_bgr = draw_coordinate_frame(img_bgr, part2_data['R'], part2_data['t'], K)

#     if args.boxes:
#         print("-> Drawing 3D bounding boxes...")
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
#         def overlay_pose(img_array, json_file, cad_file, color_rgb):
#             # --- FIX APPLIED HERE: Extract by highest score ---
#             with open(json_file) as f: 
#                 dets = json.load(f)
#             data = max(dets, key=lambda x: x['score'])
            
#             pred_rot = np.array([data['R']])
#             pred_trans = np.array([data['t']])
#             mesh = trimesh.load_mesh(cad_file)
#             model_points = mesh.sample(2048).astype(np.float32)
#             return draw_detections(img_array, pred_rot, pred_trans, model_points, np.array([K]), color=color_rgb)

#         img_rgb = overlay_pose(img_rgb, part1_json, part1_cad, color_rgb=(255, 0, 0)) # Red Base
#         img_rgb = overlay_pose(img_rgb, part2_json, part2_cad, color_rgb=(0, 255, 0)) # Green Insert
        
#         img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

#     cv2.imwrite(args.output, img_bgr)
#     print(f"Success! Visual saved to: {args.output}")

# if __name__ == '__main__':
#     main()

import argparse
import json
import numpy as np
import cv2
import os
import sys
import trimesh
import pycocotools.mask as cocomask
from scipy.spatial.transform import Rotation as R

# --- 1. Import SAM6D Native Drawing Tools ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, 'Pose_Estimation_Model')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from draw_utils import draw_detections

# --- 2. Setup Command Line Arguments ---
def get_args():
    parser = argparse.ArgumentParser(description="Assembly Verification: Metrics & Visualization")
    parser.add_argument("--scene", type=str, required=True, help="Scene folder name (e.g., 000005)")
    parser.add_argument("--frame", type=str, required=True, help="Frame ID (e.g., 000010)")
    
    parser.add_argument("--masks", action="store_true", help="Overlay 2D segmentation masks")
    parser.add_argument("--boxes", action="store_true", help="Draw 3D bounding boxes")
    parser.add_argument("--axes", action="store_true", help="Draw 3D coordinate frames")
    parser.add_argument("--output", type=str, default="pipe_assembly_viz.png", help="Output file name")
    return parser.parse_args()

# --- 3. Helper Functions ---
def make_4x4(R_list, t_list):
    T = np.eye(4)
    T[:3, :3] = np.array(R_list).reshape(3, 3)
    T[:3, 3] = np.array(t_list)
    return T

def overlay_mask(image_bgr, json_path, color_bgr, alpha=0.5):
    with open(json_path, 'r') as f:
        dets = json.load(f)
    best_det = max(dets, key=lambda x: x['score'])
    
    # --- ROBUST MASK DECODING ---
    seg = best_det["segmentation"].copy()
    if isinstance(seg["counts"], str):
        seg["counts"] = seg["counts"].encode("utf-8")
    if isinstance(seg["counts"], list):
        h_size, w_size = seg["size"]
        seg = cocomask.frPyObjects(seg, h_size, w_size)
    mask = cocomask.decode(seg).astype(bool)
    # ----------------------------
    
    for c in range(3):
        image_bgr[:, :, c] = np.where(mask,
                                      image_bgr[:, :, c] * (1 - alpha) + alpha * color_bgr[c],
                                      image_bgr[:, :, c])
    return image_bgr

def draw_coordinate_frame(img_bgr, R_mat, t_vec, K, length_mm=60.0):
    points_3d = np.array([
        [0.0, 0.0, 0.0],          
        [length_mm, 0.0, 0.0],    
        [0.0, length_mm, 0.0],    
        [0.0, 0.0, length_mm]     
    ])

    R_matrix = np.array(R_mat).reshape(3, 3)
    t_vector = np.array(t_vec).reshape(3, 1)
    points_3d_transformed = (R_matrix @ points_3d.T) + t_vector

    points_2d = K @ points_3d_transformed
    points_2d = (points_2d[:2, :] / points_2d[2, :]).T.astype(int)

    origin = tuple(points_2d[0])
    cv2.line(img_bgr, origin, tuple(points_2d[1]), (0, 0, 255), 3) # X - Red
    cv2.line(img_bgr, origin, tuple(points_2d[2]), (0, 255, 0), 3) # Y - Green
    cv2.line(img_bgr, origin, tuple(points_2d[3]), (255, 0, 0), 3) # Z - Blue
    return img_bgr

# --- 4. Main Execution ---
def main():
    args = get_args()
    
    if not (args.masks or args.boxes or args.axes):
        print("Please specify what to draw! Use --masks, --boxes, or --axes.")
        return

    base_path = f"Data/BOP/pipeconnector/test/{args.scene}/"
    FRAME_ID = args.frame
    rgb_path = base_path + f"rgb/{FRAME_ID}.png"
    cam_path = base_path + "scene_camera.json"

    # PEM paths for 3D Poses
    part1_pem_json = base_path + "outputs_part1/sam6d_results/detection_pem.json"
    part2_pem_json = base_path + "outputs_part2/sam6d_results/detection_pem.json"
    
    # ISM paths for 2D Masks
    part1_ism_json = base_path + "outputs_part1/sam6d_results/detection_ism.json"
    part2_ism_json = base_path + "outputs_part2/sam6d_results/detection_ism.json"
    
    part1_cad  = "Data/BOP/pipeconnector/models/obj_000001.ply"
    part2_cad  = "Data/BOP/pipeconnector/models/obj_000002.ply"

    img_bgr = cv2.imread(rgb_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {rgb_path}")
        return

    cam_data_all = json.load(open(cam_path))
    cam_key = list(cam_data_all.keys())[0] 
    K = np.array(cam_data_all[cam_key]['cam_K']).reshape(3, 3)

    with open(part1_pem_json) as f: 
        part1_data = max(json.load(f), key=lambda x: x['score'])
    with open(part2_pem_json) as f: 
        part2_data = max(json.load(f), key=lambda x: x['score'])

    # ==========================================
    # SECTION A: Calculate Relative Distances
    # ==========================================
    T_cam_part1 = make_4x4(part1_data['R'], part1_data['t'])
    T_cam_part2 = make_4x4(part2_data['R'], part2_data['t'])

    T_part1_part2 = np.linalg.inv(T_cam_part1) @ T_cam_part2

    t_rel = T_part1_part2[:3, 3]
    R_rel_matrix = T_part1_part2[:3, :3]
    r_euler = R.from_matrix(R_rel_matrix).as_euler('xyz', degrees=True)

    print(f"\n--- Assembly Verification Metrics (Scene {args.scene}) ---")
    print("Target: Insert (Part 2) relative to Base (Part 1)\n")
    print(f"1. Relative Position Offset (mm):")
    print(f"   X (Left/Right) : {t_rel[0]:.1f} mm")
    print(f"   Y (Up/Down)    : {t_rel[1]:.1f} mm")
    print(f"   Z (Front/Back) : {t_rel[2]:.1f} mm")
    print(f"   Total Distance : {np.linalg.norm(t_rel):.1f} mm\n")

    # ==========================================
    # SECTION B: Drawing the Visualization
    # ==========================================
    if args.masks:
        print("-> Painting segmentation masks...")
        img_bgr = overlay_mask(img_bgr, part1_ism_json, color_bgr=(0, 0, 255)) 
        img_bgr = overlay_mask(img_bgr, part2_ism_json, color_bgr=(0, 255, 0)) 

    if args.axes:
        print("-> Drawing 3D coordinate axes...")
        img_bgr = draw_coordinate_frame(img_bgr, part1_data['R'], part1_data['t'], K)
        img_bgr = draw_coordinate_frame(img_bgr, part2_data['R'], part2_data['t'], K)

    if args.boxes:
        print("-> Drawing 3D bounding boxes...")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        def overlay_pose(img_array, json_file, cad_file, color_rgb):
            with open(json_file) as f: 
                dets = json.load(f)
            data = max(dets, key=lambda x: x['score'])
            pred_rot = np.array([data['R']])
            pred_trans = np.array([data['t']])
            mesh = trimesh.load_mesh(cad_file)
            model_points = mesh.sample(2048).astype(np.float32)
            return draw_detections(img_array, pred_rot, pred_trans, model_points, np.array([K]), color=color_rgb)

        img_rgb = overlay_pose(img_rgb, part1_pem_json, part1_cad, color_rgb=(255, 0, 0)) 
        img_rgb = overlay_pose(img_rgb, part2_pem_json, part2_cad, color_rgb=(0, 255, 0)) 
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ==========================================
    # SECTION C: 3D Perspective-Correct Z-Axis Dimension Overlay
    # ==========================================
    print("-> Adding 3D perspective-correct Z-Axis dimension overlay...")
    
    # 1. Grab the 3D components of the Base (Part 1)
    R1 = np.array(part1_data['R']).reshape(3, 3)
    t1 = np.array(part1_data['t']).reshape(3, 1)
    
    # 2. Isolate the local Z-axis (Blue) and local Y-axis (Green) of the Base
    z_axis_1 = R1[:, 2].reshape(3, 1) 
    y_axis_1 = R1[:, 1].reshape(3, 1)
    
    # 3. Define the two measurement points along the perfect Z-axis in 3D
    # Point A is the origin of the base
    pt_A_3d = t1
    
    # Point B is exactly t_rel[2] millimeters away strictly along the Z-axis
    pt_B_3d = t1 + (z_axis_1 * t_rel[2])
    
    # 4. Offset both points by 60mm along the local negative Y-axis 
    # This pushes the dimension line "down" relative to the object in 3D space
    offset_dist_mm = 100.0
    v_offset_3d = -y_axis_1 * offset_dist_mm
    
    pt_A_dim_3d = pt_A_3d + v_offset_3d
    pt_B_dim_3d = pt_B_3d + v_offset_3d
    
    # 5. Helper function to project 3D points to 2D screen coordinates
    def project_to_2d(pt_3d, K_matrix):
        pt_2d = K_matrix @ pt_3d
        return (int(pt_2d[0, 0] / pt_2d[2, 0]), int(pt_2d[1, 0] / pt_2d[2, 0]))

    pA = project_to_2d(pt_A_3d, K)
    pB = project_to_2d(pt_B_3d, K)
    pA_dim = project_to_2d(pt_A_dim_3d, K)
    pB_dim = project_to_2d(pt_B_dim_3d, K)

    # 6. Draw the 3D leader lines projecting outward from the object
    cv2.line(img_bgr, pA, pA_dim, (200, 200, 200), 2)
    cv2.line(img_bgr, pB, pB_dim, (200, 200, 200), 2)

    # 7. Draw the perspective-correct dimension line
    cv2.line(img_bgr, pA_dim, pB_dim, (0, 255, 255), 3)
    cv2.circle(img_bgr, pA_dim, 4, (0, 255, 255), -1)
    cv2.circle(img_bgr, pB_dim, 4, (0, 255, 255), -1)

    # 8. Add the Z-offset text aligned near the dimension line
    z_offset = abs(t_rel[2])
    dim_text = f"Z: {z_offset:.1f} mm"
    
    # Calculate text position slightly above the midpoint of the dimension line
    mid_x = int((pA_dim[0] + pB_dim[0]) / 2)
    mid_y = int((pA_dim[1] + pB_dim[1]) / 2) - 15
    
    text_size = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    text_x = mid_x - (text_size[0] // 2)
    
    # Draw black outline, then yellow text
    cv2.putText(img_bgr, dim_text, (text_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(img_bgr, dim_text, (text_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imwrite(args.output, img_bgr)
    print(f"Success! Visual saved to: {args.output}")

if __name__ == '__main__':
    main()