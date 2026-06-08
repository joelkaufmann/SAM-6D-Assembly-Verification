import argparse
import json
import os
import numpy as np
import cv2
import trimesh

def depth_to_pointcloud(depth_img, rgb_img, cam_K, depth_scale):
    # Get depth image dimensions
    h, w = depth_img.shape
    
    # Extract intrinsic camera parameters
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert depth from millimeters (or scaled units) to actual meters
    z = (depth_img.astype(np.float32) * depth_scale) / 1000.0

    # Filter out invalid depth pixels (0 means no return signal)
    valid_mask = z > 0

    # Apply pinhole camera math to unproject 2D pixels to 3D space
    x = (u[valid_mask] - cx) * z[valid_mask] / fx
    y = (v[valid_mask] - cy) * z[valid_mask] / fy
    z = z[valid_mask]

    # Stack into an (N, 3) array
    points = np.stack((x, y, z), axis=-1)
    
    # Extract corresponding colors
    # Ensure RGB image is mapped correctly (OpenCV loads as BGR by default)
    colors = rgb_img[valid_mask]
    
    return points, colors

def main():
    parser = argparse.ArgumentParser(description="Generate a colored .ply point cloud from Scene and Frame")
    parser.add_argument("--base_dir", type=str, default="Data/BOP/pipeconnector/test", help="Root directory of the dataset")
    parser.add_argument("--scene", type=int, required=True, help="Scene ID (e.g., 13 or 24)")
    parser.add_argument("--frame", type=int, required=True, help="Frame ID (e.g., 0)")
    parser.add_argument("--output", type=str, default="debug_scene.ply", help="Name of the output file")
    args = parser.parse_args()

    # Format scene and frame as 6-digit strings
    scene_str = f"{args.scene:06d}"
    frame_str = f"{args.frame:06d}"

    scene_dir = os.path.join(args.base_dir, scene_str)
    
    # Locate RGB and Depth files (checks for both png and jpg)
    depth_path = os.path.join(scene_dir, "depth", f"{frame_str}.png")
    rgb_path = os.path.join(scene_dir, "rgb", f"{frame_str}.png")
    if not os.path.exists(rgb_path):
        rgb_path = os.path.join(scene_dir, "rgb", f"{frame_str}.jpg")

    if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
        print(f"Error: Could not find RGB or Depth image at {scene_dir}")
        return

    # Locate Camera Info (Checks standard BOP vs custom cam_info)
    cam_path = os.path.join(scene_dir, "scene_camera.json")
    if not os.path.exists(cam_path):
        cam_path = os.path.join(scene_dir, "cam_info.json")

    print(f"Loading data for Scene: {scene_str}, Frame: {frame_str}...")
    
    # Load Depth and RGB
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # Load Camera Intrinsics
    with open(cam_path, 'r') as f:
        cam_data = json.load(f)
    
    # Handle BOP format vs Direct format
    if str(args.frame) in cam_data:
        cam_info = cam_data[str(args.frame)]
    else:
        cam_info = cam_data

    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = float(cam_info['depth_scale'])

    print("Unprojecting 2D depth to 3D point cloud...")
    points, colors = depth_to_pointcloud(depth_img, rgb_img, cam_K, depth_scale)

    print(f"Generated {len(points)} valid 3D points. Saving to {args.output}...")
    
    # Save using Trimesh
    point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    point_cloud.export(args.output)
    
    print("Done! You can now open the file in MeshLab.")

if __name__ == "__main__":
    main()