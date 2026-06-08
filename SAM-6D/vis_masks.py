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

    with open(args.json_path, 'r') as f:
        predictions = json.load(f)

    img = cv2.imread(args.rgb_path)
    
    if img is None:
        print(f"FAILED: Could not load image at {args.rgb_path}")
        return

    print(f"Drawing {len(predictions)} masks from {args.json_path}...")

    for pred in predictions:
        score = pred.get("score", 0)
        seg = pred["segmentation"]
        
        # Handle uncompressed integer lists
        h, w = seg['size']
        try:
            rle = mask_util.frPyObjects(seg, h, w)
        except Exception:
            rle = seg
        binary_mask = mask_util.decode(rle)
        
        # Generate random color and apply
        color = np.random.randint(50, 255, size=(3,), dtype=np.uint8).tolist()
        img[binary_mask == 1] = img[binary_mask == 1] * 0.4 + np.array(color) * 0.6
        
        # Draw bounding box and score
        bbox = [int(x) for x in pred["bbox"]]
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        label = f"Score: {score:.3f}"
        cv2.putText(img, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Save it using the parent folder name (e.g., outputs_part1) so they don't overwrite
    parent_folder = args.json_path.split('/')[-3]
    out_name = f"debug_{parent_folder}.png"
    cv2.imwrite(out_name, img)
    print(f"Success! Saved to {out_name}")

if __name__ == "__main__":
    main()