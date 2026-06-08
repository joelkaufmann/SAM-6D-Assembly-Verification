import argparse
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Exact path to detection_ism.json")
    parser.add_argument("--rgb_path", required=True, help="Exact path to the RGB image")
    args = parser.parse_args()

    # 1. Load the data
    with open(args.json_path, 'r') as f:
        predictions = json.load(f)

    img = cv2.imread(args.rgb_path)
    if img is None:
        print(f"FAILED: Could not load image at {args.rgb_path}")
        return

    if not predictions:
        print("No predictions found in the JSON.")
        return

    # 2. Find the absolute best mask (highest score)
    best_pred = max(predictions, key=lambda x: x.get("score", 0))
    print(f"Intercepted best mask! Score: {best_pred['score']:.3f}")

    # 3. Decode the SAM mask
    seg = best_pred["segmentation"]
    h, w = seg['size']
    try:
        rle = mask_util.frPyObjects(seg, h, w)
    except Exception:
        rle = seg
    sam_mask = mask_util.decode(rle)

    # Convert the SAM mask to a standard 0-255 OpenCV mask
    sam_mask_8u = (sam_mask * 255).astype(np.uint8)

    # 4. Color Slicing (HSV) - REFINED THRESHOLDS
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Bump S to 8 and V to 30 to ignore pure dark-grey shadows
    lower_red1 = np.array([0, 8, 30])
    upper_red1 = np.array([30, 255, 255])
    
    lower_red2 = np.array([120, 8, 30])
    upper_red2 = np.array([179, 255, 255])
    
    global_red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    global_red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    raw_red_mask = cv2.bitwise_or(global_red_mask1, global_red_mask2)

    # --- NEW: MORPHOLOGICAL CLEANUP ---
    # Create a small 3x3 brush
    cleanup_kernel = np.ones((3, 3), np.uint8)
    
    # "Opening" deletes thin noise lines (like the halo) and small stray pixels
    global_red_mask = cv2.morphologyEx(raw_red_mask, cv2.MORPH_OPEN, cleanup_kernel)
    # ----------------------------------

    # 5. The Subtraction Math with a Dilation Buffer
    final_nut_mask = cv2.bitwise_and(global_red_mask, sam_mask_8u)

    # Create a 5x5 pixel brush to "inflate" the nut mask
    kernel = np.ones((5, 5), np.uint8)
    fat_nut_mask = cv2.dilate(final_nut_mask, kernel, iterations=1)

    # True Subtraction
    final_pipe_mask = cv2.bitwise_and(sam_mask_8u, cv2.bitwise_not(fat_nut_mask))

    # 6. Visualization
    # Create a dark background
    output_vis = (img * 0.3).astype(np.uint8)
    
    # Overlay the Nut in bright Red
    output_vis[final_nut_mask > 0] = [0, 0, 255] # BGR Red
    
    # Overlay the Pipe in bright Cyan
    output_vis[final_pipe_mask > 0] = [255, 255, 0] # BGR Cyan

    out_name = "debug_color_split.png"
    cv2.imwrite(out_name, output_vis)
    print(f"Success! Sliced the mask based on expanded HSV color. Saved to {out_name}")

if __name__ == "__main__":
    main()