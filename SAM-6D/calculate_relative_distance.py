import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Define paths to your calculated poses
base_path = "Data/BOP/ycbv/test/000048/"
can_path = base_path + "outputs/sam6d_results/detection_pem.json"
cup_path = base_path + "outputs_obj14/sam6d_results/detection_pem.json"

# Load the JSON data
with open(can_path) as f: can_data = json.load(f)[0]
with open(cup_path) as f: cup_data = json.load(f)[0]

# 2. Helper function to build a 4x4 matrix from SAM6D's R and t arrays
def make_4x4(R_list, t_list):
    T = np.eye(4)
    T[:3, :3] = np.array(R_list).reshape(3, 3)
    T[:3, 3] = np.array(t_list)
    return T

# Build the base matrices
T_cam_can = make_4x4(can_data['R'], can_data['t'])
T_cam_cup = make_4x4(cup_data['R'], cup_data['t'])

# 3. THE MATH: Calculate Cup pose relative to Can pose
# @ is the Python operator for matrix multiplication
T_can_cup = np.linalg.inv(T_cam_can) @ T_cam_cup

# 4. Extract and Format the Metrics
# Translation (X, Y, Z in millimeters)
t_rel = T_can_cup[:3, 3]

# Rotation (Convert 3x3 to Euler Degrees)
R_rel_matrix = T_can_cup[:3, :3]
r_euler = R.from_matrix(R_rel_matrix).as_euler('xyz', degrees=True)

# 5. Print out the Assembly Verification Results
print("\n--- Assembly Verification Metrics ---")
print("Target: Red Mug relative to Master Chef Can\n")

print(f"1. Relative Position Offset (mm):")
print(f"   X (Left/Right) : {t_rel[0]:.1f} mm")
print(f"   Y (Up/Down)    : {t_rel[1]:.1f} mm")
print(f"   Z (Front/Back) : {t_rel[2]:.1f} mm")
print(f"   Total Distance : {np.linalg.norm(t_rel):.1f} mm\n")

print(f"2. Relative Rotation/Twist (Degrees):")
print(f"   Pitch (X-axis) : {r_euler[0]:.1f} deg")
print(f"   Yaw   (Y-axis) : {r_euler[1]:.1f} deg")
print(f"   Roll  (Z-axis) : {r_euler[2]:.1f} deg")
print("-------------------------------------\n")