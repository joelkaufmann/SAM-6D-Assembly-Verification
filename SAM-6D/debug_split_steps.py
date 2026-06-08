import argparse
import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--rgb_path", required=True)
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        predictions = json.load(f)

    img = cv2.imread(args.rgb_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # 1. Get the Master SAM Mask
    best_pred = max(predictions, key=lambda x: x.get("score", 0))
    seg = best_pred["segmentation"]
    
    h, w = seg['size']
    try:
        rle = mask_util.frPyObjects(seg, h, w)
    except Exception:
        rle = seg
    sam_mask_8u = mask_util.decode(rle).astype(np.uint8)

    # 2. Get the HSV Masks
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 10, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([135, 10, 40])
    upper_red2 = np.array([179, 255, 255])
    
    red_mask = cv2.bitwise_or(cv2.inRange(hsv_img, lower_red1, upper_red1), 
                              cv2.inRange(hsv_img, lower_red2, upper_red2))

    # 3. Create Visualizations
    # Image 1: What SAM saw (The Master Mask)
    sam_vis = (img * 0.3).astype(np.uint8)
    sam_vis[sam_mask_8u == 1] = img[sam_mask_8u == 1]
    cv2.putText(sam_vis, "1. SAM Master Mask", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Image 2: What the Red Filter saw
    red_vis = (img * 0.3).astype(np.uint8)
    red_vis[red_mask > 0] = [0, 0, 255] # Highlight detected red in pure blue/red
    cv2.putText(red_vis, "2. Raw Red HSV Filter", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Image 3: The Final Subtraction (Bitwise AND)
    final_nut = cv2.bitwise_and(red_mask, sam_mask_8u)
    final_vis = (img * 0.3).astype(np.uint8)
    final_vis[final_nut > 0] = [0, 255, 0] # Highlight final in Green
    cv2.putText(final_vis, "3. Final Nut Mask (Intersection)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Stack them horizontally for easy viewing
    combined = np.hstack((sam_vis, red_vis, final_vis))
    
    # Resize to fit on screen
    h, w = combined.shape[:2]
    combined = cv2.resize(combined, (int(w * 0.5), int(h * 0.5)))

    out_name = "debug_xray.png"
    cv2.imwrite(out_name, combined)
    print(f"Saved X-Ray visualization to {out_name}")

if __name__ == "__main__":
    main()