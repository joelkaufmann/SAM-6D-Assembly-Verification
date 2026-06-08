import argparse
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
import os
import shutil
from PIL import Image
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation

def visualize(rgb_path, detections, save_path="tmp.png"):
    """Replicates the exact pipeline visualization from run_inference_custom.py"""
    rgb = Image.open(rgb_path).convert("RGB")
    img = np.array(rgb).copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = -1.0
    best_det = detections[0]
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    # Robust Mask Decoding
    seg = best_det["segmentation"].copy()
    if isinstance(seg["counts"], str):
        seg["counts"] = seg["counts"].encode("utf-8")
    if isinstance(seg["counts"], list):
        h_size, w_size = seg["size"]
        seg = mask_util.frPyObjects(seg, h_size, w_size)
        
    mask = mask_util.decode(seg).astype(bool)

    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    # obj_id = best_det["category_id"]
    # temp_id = obj_id - 
    temp_id = 0

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img_pil = Image.fromarray(np.uint8(img))
    
    # concat side by side in PIL
    rgb_arr = np.array(rgb)
    concat = Image.new('RGB', (img_pil.width + rgb_arr.shape[1], img_pil.height))
    concat.paste(rgb, (0, 0))
    concat.paste(img_pil, (rgb_arr.shape[1], 0))
    concat.save(save_path)

def main():
    parser = argparse.ArgumentParser(description="Standalone post-processing script to split SAM masks by color.")
    parser.add_argument("--json_path", required=True, help="Path to detection_ism.json")
    parser.add_argument("--rgb_path", required=True, help="Path to the RGB image")
    parser.add_argument("--target", required=True, choices=['nut', 'pipe'], help="Which part to isolate")
    args = parser.parse_args()

    # ==========================================
    # 1. THE BACKUP SYSTEM (ALLOWS MULTIPLE TRIES)
    # ==========================================
    raw_json_path = args.json_path.replace(".json", "_RAW.json")
    out_dir = os.path.dirname(args.json_path)
    vis_path = os.path.join(out_dir, "vis_ism.png")
    raw_vis_path = vis_path.replace(".png", "_RAW.png")

    # If first time running, backup the original SAM files
    if not os.path.exists(raw_json_path):
        print("--> Backing up original SAM JSON and Visualization...")
        shutil.copy(args.json_path, raw_json_path)
        if os.path.exists(vis_path):
            shutil.copy(vis_path, raw_vis_path)
        
    # ALWAYS load from the pristine RAW backup
    with open(raw_json_path, 'r') as f:
        predictions = json.load(f)
        
    if not predictions:
        print("No predictions to process.")
        return

    # ==========================================
    # 2. EXTRACT SAM MASK
    # ==========================================
    img = cv2.imread(args.rgb_path)
    if img is None:
        print(f"FAILED: Could not load image at {args.rgb_path}")
        return

    best_pred = max(predictions, key=lambda x: x.get("score", 0))
    seg = best_pred["segmentation"]
    
    h, w = seg['size']
    try:
        rle = mask_util.frPyObjects(seg, h, w)
    except Exception:
        rle = seg
    sam_mask = mask_util.decode(rle)
    sam_mask_8u = (sam_mask * 255).astype(np.uint8)

    # ==========================================
    # 3. APPLY YOUR EXACT COLOR SPLITTING LOGIC
    # ==========================================
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 8, 30])
    upper_red1 = np.array([30, 255, 255])
    
    lower_red2 = np.array([120, 8, 30])
    upper_red2 = np.array([179, 255, 255])
    
    global_red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    global_red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    raw_red_mask = cv2.bitwise_or(global_red_mask1, global_red_mask2)

    # Morphological Cleanup
    cleanup_kernel = np.ones((3, 3), np.uint8)
    global_red_mask = cv2.morphologyEx(raw_red_mask, cv2.MORPH_OPEN, cleanup_kernel)

    # The Subtraction Math with Dilation Buffer
    final_nut_mask = cv2.bitwise_and(global_red_mask, sam_mask_8u)

    kernel = np.ones((5, 5), np.uint8)
    fat_nut_mask = cv2.dilate(final_nut_mask, kernel, iterations=1)
    final_pipe_mask = cv2.bitwise_and(sam_mask_8u, cv2.bitwise_not(fat_nut_mask))

    # ==========================================
    # 4. SELECT TARGET & UPDATE RLE
    # ==========================================
    if args.target == 'nut':
        final_nut_mask[final_nut_mask > 0] = 1 # Binarize
        final_mask = final_nut_mask
        category_id = 1
    elif args.target == 'pipe':
        final_pipe_mask[final_pipe_mask > 0] = 1 # Binarize
        final_mask = final_pipe_mask
        category_id = 2

    # Convert back to COCO RLE Format
    fortran_mask = np.asfortranarray(final_mask)
    new_rle = mask_util.encode(fortran_mask)
    new_rle['counts'] = new_rle['counts'].decode('utf-8') 
    new_bbox = mask_util.toBbox(new_rle).tolist()
    
    # Overwrite the prediction entry
    new_pred = best_pred.copy()
    new_pred['segmentation'] = new_rle
    new_pred['bbox'] = new_bbox
    new_pred['category_id'] = category_id
    new_pred['score'] = 0.99 
    
    # ==========================================
    # 5. SAVE JSON AND UPDATE VISUALIZATION
    # ==========================================
    with open(args.json_path, 'w') as f:
        json.dump([new_pred], f)
        
    visualize(args.rgb_path, [new_pred], vis_path)
    
    print(f"[{args.target.upper()}] Success! Mask isolated based on HSV logic.")
    print(f"--> Overwrote: {args.json_path}")
    print(f"--> Updated: {vis_path}")

if __name__ == "__main__":
    main()