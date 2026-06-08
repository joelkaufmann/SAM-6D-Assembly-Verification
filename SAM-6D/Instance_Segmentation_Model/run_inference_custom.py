import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    
    logging.info("Initializing template")
    template_dir = os.path.join(output_dir, 'templates')
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data
    
    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    # compute the geometric score
    batch = batch_input_data(depth_path, cam_path, device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )

    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
    detections.to_numpy()
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    vis_img = visualize(rgb, detections, f"{output_dir}/sam6d_results/vis_ism.png")
    vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
    run_inference(
        args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
        stability_score_thresh=args.stability_score_thresh,
    )

# import os, sys
# import numpy as np
# import shutil
# from tqdm import tqdm
# import time
# import torch
# from PIL import Image
# import logging
# import os.path as osp
# from hydra import initialize, compose
# # set level logging
# logging.basicConfig(level=logging.INFO)
# import trimesh
# from hydra.utils import instantiate
# import argparse
# import glob
# from omegaconf import DictConfig, OmegaConf
# from torchvision.utils import save_image
# import torchvision.transforms as T
# import cv2
# import imageio.v2 as imageio # Fixed Deprecation Warning
# import distinctipy
# import json
# import pycocotools.mask as mask_util
# from skimage.feature import canny
# from skimage.morphology import binary_dilation

# from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
# from utils.bbox_utils import CropResizePad
# from model.utils import Detections, convert_npz_to_json
# from model.loss import Similarity
# from utils.inout import load_json, save_json_bop23

# inv_rgb_transform = T.Compose(
#         [
#             T.Normalize(
#                 mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
#             ),
#         ]
#     )

# def visualize(rgb, detections, save_path="tmp.png"):
#     img = rgb.copy()
#     gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
#     img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
#     colors = distinctipy.get_colors(len(detections))
#     alpha = 0.33

#     best_score = -1.0
#     best_det = detections[0]
#     for mask_idx, det in enumerate(detections):
#         if best_score < det['score']:
#             best_score = det['score']
#             best_det = detections[mask_idx]

#     # --- FIXED: Robust Mask Decoding ---
#     seg = best_det["segmentation"].copy()
    
#     # If the hack compressed it to a string, encode it back to bytes for pycocotools
#     if isinstance(seg["counts"], str):
#         seg["counts"] = seg["counts"].encode("utf-8")
        
#     # If SAM left it as an uncompressed integer list, format it
#     if isinstance(seg["counts"], list):
#         h_size, w_size = seg["size"]
#         seg = mask_util.frPyObjects(seg, h_size, w_size)
        
#     # Decode to binary mask
#     mask = mask_util.decode(seg).astype(bool)
#     # -----------------------------------

#     edge = canny(mask)
#     edge = binary_dilation(edge, np.ones((2, 2)))
#     obj_id = best_det["category_id"]
#     temp_id = obj_id - 1

#     r = int(255*colors[temp_id][0])
#     g = int(255*colors[temp_id][1])
#     b = int(255*colors[temp_id][2])
#     img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
#     img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
#     img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
#     img[edge, :] = 255
    
#     img = Image.fromarray(np.uint8(img))
#     img.save(save_path)
#     prediction = Image.open(save_path)
    
#     # concat side by side in PIL
#     img = np.array(img)
#     concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
#     concat.paste(rgb, (0, 0))
#     concat.paste(prediction, (img.shape[1], 0))
#     return concat

# def batch_input_data(depth_path, cam_path, device):
#     batch = {}
#     cam_info = load_json(cam_path)
#     depth = np.array(imageio.imread(depth_path)).astype(np.int32)
#     cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
#     depth_scale = np.array(cam_info['depth_scale'])

#     batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
#     batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
#     batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
#     return batch

# def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh, target_color):
#     with initialize(version_base=None, config_path="configs"):
#         cfg = compose(config_name='run_inference.yaml')

#     if segmentor_model == "sam":
#         with initialize(version_base=None, config_path="configs/model"):
#             cfg.model = compose(config_name='ISM_sam.yaml')
#         cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
#     elif segmentor_model == "fastsam":
#         with initialize(version_base=None, config_path="configs/model"):
#             cfg.model = compose(config_name='ISM_fastsam.yaml')
#     else:
#         raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

#     logging.info("Initializing model")
#     model = instantiate(cfg.model)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.descriptor_model.model = model.descriptor_model.model.to(device)
#     model.descriptor_model.model.device = device
#     # if there is predictor in the model, move it to device
#     if hasattr(model.segmentor_model, "predictor"):
#         model.segmentor_model.predictor.model = (
#             model.segmentor_model.predictor.model.to(device)
#         )
#     else:
#         model.segmentor_model.model.setup_model(device=device, verbose=True)
#     logging.info(f"Moving models to {device} done!")
        
    
#     logging.info("Initializing template")
#     template_dir = os.path.join(output_dir, 'templates')
#     num_templates = len(glob.glob(f"{template_dir}/*.npy"))
#     boxes, masks, templates = [], [], []
#     for idx in range(num_templates):
#         image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
#         mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
#         boxes.append(mask.getbbox())

#         image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
#         mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
#         image = image * mask[:, :, None]
#         templates.append(image)
#         masks.append(mask.unsqueeze(-1))
        
#     templates = torch.stack(templates).permute(0, 3, 1, 2)
#     masks = torch.stack(masks).permute(0, 3, 1, 2)
#     boxes = torch.tensor(np.array(boxes))
    
#     processing_config = OmegaConf.create(
#         {
#             "image_size": 224,
#         }
#     )
#     proposal_processor = CropResizePad(processing_config.image_size)
#     templates = proposal_processor(images=templates, boxes=boxes).to(device)
#     masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

#     model.ref_data = {}
#     model.ref_data["descriptors"] = model.descriptor_model.compute_features(
#                     templates, token_name="x_norm_clstoken"
#                 ).unsqueeze(0).data
#     model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
#                     templates, masks_cropped[:, 0, :, :]
#                 ).unsqueeze(0).data
    
#     # run inference
#     rgb = Image.open(rgb_path).convert("RGB")
#     detections = model.segmentor_model.generate_masks(np.array(rgb))
#     detections = Detections(detections)
#     query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

#     # matching descriptors
#     (
#         idx_selected_proposals,
#         pred_idx_objects,
#         semantic_score,
#         best_template,
#     ) = model.compute_semantic_score(query_decriptors)

#     # update detections
#     detections.filter(idx_selected_proposals)
#     query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

#     # compute the appearance score
#     appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

#     # compute the geometric score
#     batch = batch_input_data(depth_path, cam_path, device)
#     template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
#     template_poses[:, :3, 3] *= 0.4
#     poses = torch.tensor(template_poses).to(torch.float32).to(device)
#     model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

#     mesh = trimesh.load_mesh(cad_path)
#     model_points = mesh.sample(2048).astype(np.float32) / 1000.0
#     model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
#     image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

#     geometric_score, visible_ratio = model.compute_geometric_score(
#         image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
#         )

#     # final score
#     final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

#     detections.add_attribute("scores", final_score)
#     detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
#     detections.to_numpy()
#     save_path = f"{output_dir}/sam6d_results/detection_ism"
#     detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
#     detections_json = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
#     save_json_bop23(save_path+".json", detections_json)
    
#     # ==========================================
#     # POST-PROCESSING: RELATIVE COLOR RATIO HACK
#     # ==========================================
#     if target_color is not None:
#         logging.info(f"Applying relative color ratio hack for target: {target_color}")
#         json_path = save_path + ".json"
        
#         with open(json_path, 'r') as f:
#             preds = json.load(f)
            
#         if preds:
#             best_pred = max(preds, key=lambda x: x.get("score", 0))
#             seg = best_pred["segmentation"]
            
#             h_size, w_size = seg['size']
#             try:
#                 rle = mask_util.frPyObjects(seg, h_size, w_size)
#             except Exception:
#                 rle = seg
#             sam_mask_8u = mask_util.decode(rle).astype(np.uint8)

#             img_cv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

#             # Convert to float32 to prevent 8-bit overflow during multiplication
#             img_float = img_cv.astype(np.float32)
#             b, g, r = cv2.split(img_float)

#             # A pixel is "red/pink" if the Red channel is at least 3% stronger than Green and Blue.
#             # r > 20 ensures we ignore pitch-black shadows.
#             ratio_mask = (r > (g * 1.03)) & (r > (b * 1.03)) & (r > 20)
            
#             # Convert boolean array to standard OpenCV 0-255 mask
#             raw_red_mask = (ratio_mask * 255).astype(np.uint8)

#             # --- MORPHOLOGICAL CLEANUP ---
#             # "Opening" deletes thin noise lines and stray pixels to prevent bleeding
#             cleanup_kernel = np.ones((3, 3), np.uint8)
#             clean_red_mask = cv2.morphologyEx(raw_red_mask, cv2.MORPH_OPEN, cleanup_kernel)
            
#             final_nut_mask = cv2.bitwise_and(clean_red_mask, sam_mask_8u)
#             final_nut_mask[final_nut_mask > 0] = 1 # Binarize for RLE

#             if target_color == 'nut':
#                 final_mask = final_nut_mask
#                 category_id = 1
#             elif target_color == 'pipe':
#                 # Dilation Subtraction Buffer
#                 kernel = np.ones((5, 5), np.uint8)
#                 fat_nut_mask = cv2.dilate(final_nut_mask, kernel, iterations=1)
#                 final_pipe_mask = cv2.bitwise_and(sam_mask_8u, cv2.bitwise_not(fat_nut_mask))
#                 final_pipe_mask[final_pipe_mask > 0] = 1
#                 final_mask = final_pipe_mask
#                 category_id = 2
#             else:
#                 raise ValueError("target_color must be 'nut' or 'pipe'")
            
#             # Convert modified mask back to COCO RLE Format
#             fortran_mask = np.asfortranarray(final_mask)
#             new_rle = mask_util.encode(fortran_mask)
#             new_rle['counts'] = new_rle['counts'].decode('utf-8') 
#             new_bbox = mask_util.toBbox(new_rle).tolist()
            
#             # Overwrite the prediction entry
#             new_pred = best_pred.copy()
#             new_pred['segmentation'] = new_rle
#             new_pred['bbox'] = new_bbox
#             new_pred['category_id'] = category_id
#             new_pred['score'] = 0.99  # Force PEM to trust the new mask
            
#             # Overwrite the JSON file
#             with open(json_path, 'w') as f:
#                 json.dump([new_pred], f)
                
#             # --- UPDATE THE VISUALIZATION ---
#             vis_img = visualize(rgb, [new_pred], f"{output_dir}/sam6d_results/vis_ism.png")
#             vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")
            
#             logging.info(f"Successfully overwrote JSON and updated vis_ism.png for {target_color}.")
            
#     else:
#         # If not using the hack, just visualize the original output
#         vis_img = visualize(rgb, detections_json, f"{output_dir}/sam6d_results/vis_ism.png")
#         vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
#     parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
#     parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
#     parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
#     parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
#     parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
#     parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
#     parser.add_argument("--target_color", default=None, choices=['nut', 'pipe'], help="Apply color split hack to isolate part")
#     args = parser.parse_args()
#     os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
#     run_inference(
#         args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
#         stability_score_thresh=args.stability_score_thresh, target_color=args.target_color
#     )


# import os, sys
# import numpy as np
# import shutil
# from tqdm import tqdm
# import time
# import torch
# from PIL import Image
# import logging
# import os.path as osp
# from hydra import initialize, compose
# # set level logging
# logging.basicConfig(level=logging.INFO)
# import trimesh
# from hydra.utils import instantiate
# import argparse
# import glob
# from omegaconf import DictConfig, OmegaConf
# from torchvision.utils import save_image
# import torchvision.transforms as T
# import cv2
# import imageio.v2 as imageio # Fixed Deprecation Warning
# import distinctipy
# import json
# import pycocotools.mask as mask_util
# from skimage.feature import canny
# from skimage.morphology import binary_dilation

# from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
# from utils.bbox_utils import CropResizePad
# from model.utils import Detections, convert_npz_to_json
# from model.loss import Similarity
# from utils.inout import load_json, save_json_bop23

# inv_rgb_transform = T.Compose(
#         [
#             T.Normalize(
#                 mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
#             ),
#         ]
#     )

# def visualize(rgb, detections, save_path="tmp.png"):
#     img = rgb.copy()
#     gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
#     img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
#     colors = distinctipy.get_colors(len(detections))
#     alpha = 0.33

#     best_score = -1.0
#     best_det = detections[0]
#     for mask_idx, det in enumerate(detections):
#         if best_score < det['score']:
#             best_score = det['score']
#             best_det = detections[mask_idx]

#     # --- FIXED: Robust Mask Decoding ---
#     seg = best_det["segmentation"].copy()
    
#     # If the hack compressed it to a string, encode it back to bytes for pycocotools
#     if isinstance(seg["counts"], str):
#         seg["counts"] = seg["counts"].encode("utf-8")
        
#     # If SAM left it as an uncompressed integer list, format it
#     if isinstance(seg["counts"], list):
#         h_size, w_size = seg["size"]
#         seg = mask_util.frPyObjects(seg, h_size, w_size)
        
#     # Decode to binary mask
#     mask = mask_util.decode(seg).astype(bool)
#     # -----------------------------------

#     edge = canny(mask)
#     edge = binary_dilation(edge, np.ones((2, 2)))
#     obj_id = best_det["category_id"]
#     temp_id = obj_id - 1

#     r = int(255*colors[temp_id][0])
#     g = int(255*colors[temp_id][1])
#     b = int(255*colors[temp_id][2])
#     img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
#     img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
#     img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
#     img[edge, :] = 255
    
#     img = Image.fromarray(np.uint8(img))
#     img.save(save_path)
#     prediction = Image.open(save_path)
    
#     # concat side by side in PIL
#     img = np.array(img)
#     concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
#     concat.paste(rgb, (0, 0))
#     concat.paste(prediction, (img.shape[1], 0))
#     return concat

# def batch_input_data(depth_path, cam_path, device):
#     batch = {}
#     cam_info = load_json(cam_path)
#     depth = np.array(imageio.imread(depth_path)).astype(np.int32)
#     cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
#     depth_scale = np.array(cam_info['depth_scale'])

#     batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
#     batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
#     batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
#     return batch

# def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh, target_color):
#     with initialize(version_base=None, config_path="configs"):
#         cfg = compose(config_name='run_inference.yaml')

#     if segmentor_model == "sam":
#         with initialize(version_base=None, config_path="configs/model"):
#             cfg.model = compose(config_name='ISM_sam.yaml')
#         cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
#     elif segmentor_model == "fastsam":
#         with initialize(version_base=None, config_path="configs/model"):
#             cfg.model = compose(config_name='ISM_fastsam.yaml')
#     else:
#         raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

#     logging.info("Initializing model")
#     model = instantiate(cfg.model)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.descriptor_model.model = model.descriptor_model.model.to(device)
#     model.descriptor_model.model.device = device
#     # if there is predictor in the model, move it to device
#     if hasattr(model.segmentor_model, "predictor"):
#         model.segmentor_model.predictor.model = (
#             model.segmentor_model.predictor.model.to(device)
#         )
#     else:
#         model.segmentor_model.model.setup_model(device=device, verbose=True)
#     logging.info(f"Moving models to {device} done!")
        
    
#     logging.info("Initializing template")
#     template_dir = os.path.join(output_dir, 'templates')
#     num_templates = len(glob.glob(f"{template_dir}/*.npy"))
#     boxes, masks, templates = [], [], []
#     for idx in range(num_templates):
#         image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
#         mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
#         boxes.append(mask.getbbox())

#         image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
#         mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
#         image = image * mask[:, :, None]
#         templates.append(image)
#         masks.append(mask.unsqueeze(-1))
        
#     templates = torch.stack(templates).permute(0, 3, 1, 2)
#     masks = torch.stack(masks).permute(0, 3, 1, 2)
#     boxes = torch.tensor(np.array(boxes))
    
#     processing_config = OmegaConf.create(
#         {
#             "image_size": 224,
#         }
#     )
#     proposal_processor = CropResizePad(processing_config.image_size)
#     templates = proposal_processor(images=templates, boxes=boxes).to(device)
#     masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

#     model.ref_data = {}
#     model.ref_data["descriptors"] = model.descriptor_model.compute_features(
#                     templates, token_name="x_norm_clstoken"
#                 ).unsqueeze(0).data
#     model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
#                     templates, masks_cropped[:, 0, :, :]
#                 ).unsqueeze(0).data
    
#     # run inference
#     rgb = Image.open(rgb_path).convert("RGB")
#     detections = model.segmentor_model.generate_masks(np.array(rgb))
#     detections = Detections(detections)
#     query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

#     # matching descriptors
#     (
#         idx_selected_proposals,
#         pred_idx_objects,
#         semantic_score,
#         best_template,
#     ) = model.compute_semantic_score(query_decriptors)

#     # update detections
#     detections.filter(idx_selected_proposals)
#     query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

#     # compute the appearance score
#     appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

#     # compute the geometric score
#     batch = batch_input_data(depth_path, cam_path, device)
#     template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
#     template_poses[:, :3, 3] *= 0.4
#     poses = torch.tensor(template_poses).to(torch.float32).to(device)
#     model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

#     mesh = trimesh.load_mesh(cad_path)
#     model_points = mesh.sample(2048).astype(np.float32) / 1000.0
#     model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
#     image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

#     geometric_score, visible_ratio = model.compute_geometric_score(
#         image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
#         )

#     # final score
#     final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

#     detections.add_attribute("scores", final_score)
#     detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
#     detections.to_numpy()
#     save_path = f"{output_dir}/sam6d_results/detection_ism"
#     detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
#     detections_json = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
#     save_json_bop23(save_path+".json", detections_json)
    
#     # ==========================================
#     # POST-PROCESSING: THE COLOR SPLIT HACK
#     # ==========================================
#     if target_color is not None:
#         logging.info(f"Applying color split hack for target: {target_color}")
#         json_path = save_path + ".json"
        
#         with open(json_path, 'r') as f:
#             preds = json.load(f)
            
#         if preds:
#             best_pred = max(preds, key=lambda x: x.get("score", 0))
#             seg = best_pred["segmentation"]
            
#             h_size, w_size = seg['size']
#             try:
#                 rle = mask_util.frPyObjects(seg, h_size, w_size)
#             except Exception:
#                 rle = seg
#             sam_mask_8u = mask_util.decode(rle).astype(np.uint8)

#             img_cv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
#             hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

#             # Isolate Red/Pink Powder
#             lower_red1 = np.array([0, 10, 40])
#             upper_red1 = np.array([25, 255, 255])
#             lower_red2 = np.array([135, 10, 40])
#             upper_red2 = np.array([179, 255, 255])
            
#             red_mask = cv2.bitwise_or(cv2.inRange(hsv_img, lower_red1, upper_red1), 
#                                       cv2.inRange(hsv_img, lower_red2, upper_red2))
            
#             final_nut_mask = cv2.bitwise_and(red_mask, sam_mask_8u)
#             final_nut_mask[final_nut_mask > 0] = 1

#             if target_color == 'nut':
#                 final_mask = final_nut_mask
#                 category_id = 1
#             elif target_color == 'pipe':
#                 # Dilation Subtraction Buffer
#                 kernel = np.ones((5, 5), np.uint8)
#                 fat_nut_mask = cv2.dilate(final_nut_mask, kernel, iterations=1)
#                 final_pipe_mask = cv2.bitwise_and(sam_mask_8u, cv2.bitwise_not(fat_nut_mask))
#                 final_pipe_mask[final_pipe_mask > 0] = 1
#                 final_mask = final_pipe_mask
#                 category_id = 2
#             else:
#                 raise ValueError("target_color must be 'nut' or 'pipe'")
            
#             # Convert modified mask back to COCO RLE Format
#             fortran_mask = np.asfortranarray(final_mask)
#             new_rle = mask_util.encode(fortran_mask)
#             new_rle['counts'] = new_rle['counts'].decode('utf-8') 
#             new_bbox = mask_util.toBbox(new_rle).tolist()
            
#             # Overwrite the prediction entry
#             new_pred = best_pred.copy()
#             new_pred['segmentation'] = new_rle
#             new_pred['bbox'] = new_bbox
#             new_pred['category_id'] = category_id
#             new_pred['score'] = 0.99  # Force PEM to trust the new mask
            
#             # Overwrite the JSON file
#             with open(json_path, 'w') as f:
#                 json.dump([new_pred], f)
                
#             # --- UPDATE THE VISUALIZATION ---
#             vis_img = visualize(rgb, [new_pred], f"{output_dir}/sam6d_results/vis_ism.png")
#             vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")
            
#             logging.info(f"Successfully overwrote JSON and updated vis_ism.png for {target_color}.")
            
#     else:
#         # If not using the hack, just visualize the original output
#         vis_img = visualize(rgb, detections_json, f"{output_dir}/sam6d_results/vis_ism.png")
#         vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
#     parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
#     parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
#     parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
#     parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
#     parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
#     parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
#     parser.add_argument("--target_color", default=None, choices=['nut', 'pipe'], help="Apply color split hack to isolate part")
#     args = parser.parse_args()
#     os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
#     run_inference(
#         args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
#         stability_score_thresh=args.stability_score_thresh, target_color=args.target_color
#     )