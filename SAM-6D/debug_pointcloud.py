import json
import numpy as np
import cv2
import trimesh
import pycocotools.mask as cocomask
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Extract Masked Point Cloud & 2D Overlay")
    parser.add_argument("--scene", type=str, required=True, help="Scene folder (e.g., 000010)")
    parser.add_argument("--frame", type=str, required=True, help="Frame ID (e.g., 000124)")
    parser.add_argument("--part", type=str, required=True, help="Part folder (outputs_part1 or outputs_part2)")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Paths
    scene_dir = f"Data/BOP/pipeconnector/test/{args.scene}"
    rgb_path = f"{scene_dir}/rgb/{args.frame}.png"
    depth_path = f"{scene_dir}/depth/{args.frame}.png"
    cam_path = f"{scene_dir}/camera_custom.json"
    ism_path = f"{scene_dir}/{args.part}/sam6d_results/detection_ism.json"

    print(f"Loading data for Scene {args.scene}, Frame {args.frame}, {args.part}...")

    # 1. Load Camera Matrix
    with open(cam_path) as f:
        cam_info = json.load(f)
    K = np.array(cam_info['cam_K']).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    depth_scale = cam_info.get('depth_scale', 1.0)

    # 2. Load 
    rgb_bgr = cv2.imread(rgb_path) # Keep BGR for saving the 2D overlay later
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB) # Use RGB for 3D point colors
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) * depth_scale / 1000.0

    # 3. Load the Highest-Scoring Mask
    with open(ism_path) as f:
        dets = json.load(f)
    best_det = max(dets, key=lambda x: x['score'])
    seg = best_det['segmentation']

    h, w = seg['size']
    try:
        rle = cocomask.frPyObjects(seg, h, w)
    except:
        rle = seg
    mask = cocomask.decode(rle)

    # ==========================================
    # NEW: Save 2D Mask Overlay Image (Contour Method)
    # ==========================================
    overlay_img = rgb_bgr.copy()
    
    # 1. Darken the background so the segmented object pops
    # This leaves the object's pixels 100% untouched for inspection
    overlay_img[mask == 0] = (overlay_img[mask == 0] * 0.3).astype(np.uint8)
    
    # 2. Find the exact boundary edges of the 2D mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Draw a sharp 1-pixel red outline exactly on the mask boundary
    cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 1) 
    
    overlay_filename = f"debug_overlay_{args.part}.jpg"
    cv2.imwrite(overlay_filename, overlay_img)
    print(f"-> Saved 2D Overlay: {overlay_filename}")

    # # ==========================================
    # # NEW: Save 2D Mask Overlay Image
    # # ==========================================
    # overlay_img = rgb_bgr.copy()
    # # Paint the masked area with a 50% transparent green tint
    # overlay_img[mask > 0] = overlay_img[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # overlay_filename = f"debug_overlay_{args.part}.jpg"
    # cv2.imwrite(overlay_filename, overlay_img)
    # print(f"-> Saved 2D Overlay: {overlay_filename}")

    # ==========================================
    # 4. Extract 3D Points inside the Mask (WITH AI FIXES)
    # ==========================================
    valid_pixels = np.logical_and(mask > 0, depth > 0)
    v, u = np.where(valid_pixels)

    z = depth[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_3d = np.stack((x, y, z), axis=-1)
    colors_rgb = rgb_rgb[v, u]

    # --- NEW: Match the PEM Z-Clipping (Tail Amputation) ---
    if len(points_3d) > 0:
        closest_z = np.min(points_3d[:, 2])
        max_allowed_z = closest_z + 0.142  # 14.2 cm cutoff behind the front face
        valid_flag = points_3d[:, 2] < max_allowed_z
        
        points_3d = points_3d[valid_flag]
        colors_rgb = colors_rgb[valid_flag]

    # --- NEW: Match the PEM Memory Subsampling ---
    max_points = 10000
    if len(points_3d) > max_points:
        idx = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[idx]
        colors_rgb = colors_rgb[idx]
    # --------------------------------------------------------

    # 5. Save the Point Cloud
    cloud_filename = f"debug_cloud_{args.part}.ply"
    cloud = trimesh.PointCloud(points_3d, colors=colors_rgb)
    cloud.export(cloud_filename)
    
    print(f"-> Saved 3D Cloud ({len(points_3d)} points): {cloud_filename}")
    print("SUCCESS!")

if __name__ == "__main__":
    main()