"""
Author: Gautham Ramkumar, Yoga Srinivas Reddy Kasireddy, Sai Vamsi Rithvik Allanka
Date: 2025-12-11
CS7180 - Advanced Perception
"""

import os
import glob
import argparse
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# ==========================================
# CONFIGURATION (Based on Preliminary Review)
# ------------------------------------------
# These constants control how colors are binned and which
# detected objects are considered valid for building priors.
# ==========================================
NUM_BINS = 8  # Reduced to 8 for stability (coarser, more robust bins)
BIN_SIZE = 256 // NUM_BINS
BLACK_THRESHOLD = 30   # Per-channel threshold for discarding very dark pixels
MIN_PIXEL_COUNT = 150  # Reject masks with very few pixels (likely noise)


def get_voxel_index(rgb_val):
    """
    Map RGB values (0–255) to discrete voxel indices (0–NUM_BINS-1).

    High-level idea:
    - The color space [0, 255] for each channel is divided into NUM_BINS equal
      intervals (bins).
    - Each channel value is assigned a bin index via integer division.
    - The result is a coarse "voxel" index that groups similar colors together.
    """
    return np.clip(rgb_val // BIN_SIZE, 0, NUM_BINS - 1)


def clean_mask(mask):
    """
    Clean and refine a segmentation mask.

    High-level idea:
    - Convert a soft/float mask into a binary mask.
    - Apply morphological closing to fill small holes and smooth shapes.
    - Apply a small erosion to remove noisy edges and bleed from borders.
    - Return a boolean mask that is more reliable for pixel sampling.
    """
    # 1. Threshold to binary
    mask_bin = (mask > 0.5).astype(np.uint8)
    
    # 2. Morphological Closing (Fill holes and connect small gaps)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    
    # 3. Slight Erosion (Shrink mask a bit to remove edge noise)
    mask_clean = cv2.erode(mask_clean, kernel, iterations=1)
    return mask_clean.astype(bool)


def process_dataset(data_dirs, model_path, output_path, conf_thresh=0.5):
    """
    Run YOLO segmentation over a dataset and build robust color priors per class.

    High-level pipeline:
    1. Load a segmentation model (YOLO).
    2. Collect all image file paths from the given directory/directories.
    3. For each image:
       - Run segmentation.
       - For each detected instance:
         * Align and clean the mask.
         * Extract pixels belonging to that object.
         * Filter out pixels that are too dark.
         * Reject small or noisy instances by size.
         * Optionally subsample large pixel sets for efficiency.
         * Compute:
           - Per-instance median color (for global fallback/median).
           - Per-voxel statistics: sum of colors + pixel counts for each color bin.
    4. After processing all images:
       - For each class:
         * Compute a global median color from all instance medians.
         * For each voxel:
           - Discard voxels that have very few pixels (low support).
           - Compute voxel centroid (mean color) and relative weight.
       - Assemble all information into a dictionary of priors.
    5. Save the resulting priors to a YAML file.
    """
    # Load YOLO segmentation model
    model = YOLO(model_path)
    
    # 1. Gather Images (supports multiple extensions and nested folders)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    for d in data_dirs:
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print("No images found!")
        return

    # Accumulators for voxel-level statistics:
    # class_name -> voxel_key -> {'sum': np.array(3,), 'count': int}
    class_voxels = {}
    
    # Accumulator for per-instance medians:
    # class_name -> list of median RGB colors (one per object instance)
    class_instance_medians = {}
    
    print(f"Building Robust Priors from {len(image_paths)} images...")
    
    # Iterate over all images and update statistics
    for path in tqdm(image_paths):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Run YOLO segmentation on the image
        try:
            results = model(img_rgb, conf=conf_thresh, verbose=False)[0]
        except:
            # If inference fails for some reason, skip this image
            continue
        if results.masks is None:
            # No masks / no detections
            continue

        masks = results.masks.data.cpu().numpy()
        names = results.names              # class_id -> class_name mapping
        class_ids = results.boxes.cls.cpu().numpy()

        # Process each segmentation mask / instance
        for i, mask in enumerate(masks):
            # 1. Ensure mask resolution matches the original image
            if mask.shape[:2] != img_rgb.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (img_rgb.shape[1], img_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # 2. Clean the mask using morphological operations
            mask_bool = clean_mask(mask)
            
            # 3. Extract all pixels belonging to this object
            pixels = img_rgb[mask_bool]
            if pixels.size == 0:
                continue
            
            # 4. Filter out very dark pixels using max channel threshold.
            #    This keeps colored objects even if they are somewhat dark,
            #    but discards almost-black regions.
            max_channel = np.max(pixels, axis=1)
            valid_pixels = pixels[max_channel > BLACK_THRESHOLD]
            
            # 5. Reject tiny objects (insufficient support for robust statistics)
            if len(valid_pixels) < MIN_PIXEL_COUNT:
                continue
            
            # 6. Optionally subsample large objects to a manageable number of pixels
            if len(valid_pixels) > 2000:
                indices = np.linspace(0, len(valid_pixels)-1, 2000).astype(int)
                valid_pixels = valid_pixels[indices]
            
            # Class name for this instance
            cls_name = names[int(class_ids[i])]
            if cls_name not in class_voxels:
                class_voxels[cls_name] = {}
                class_instance_medians[cls_name] = []
            
            # Store per-instance median color for later global median computation
            instance_median = np.median(valid_pixels, axis=0)
            class_instance_medians[cls_name].append(instance_median)
            
            # Quantize colors into (R,G,B) voxel indices
            r_idxs = get_voxel_index(valid_pixels[:, 0])
            g_idxs = get_voxel_index(valid_pixels[:, 1])
            b_idxs = get_voxel_index(valid_pixels[:, 2])
            
            # Pack (r_idx, g_idx, b_idx) into a single integer hash for grouping
            voxel_hashes = (
                r_idxs.astype(np.int64) * 10000 +
                g_idxs.astype(np.int64) * 100 +
                b_idxs.astype(np.int64)
            )
            unique_hashes, counts = np.unique(voxel_hashes, return_counts=True)
            
            # Accumulate color sums and counts per voxel
            for h, count in zip(unique_hashes, counts):
                r_idx = h // 10000
                g_idx = (h % 10000) // 100
                b_idx = h % 100
                key = f"{r_idx}_{g_idx}_{b_idx}"
                
                # Select pixels that fall into this voxel
                mask_for_voxel = (voxel_hashes == h)
                pixels_in_voxel = valid_pixels[mask_for_voxel]
                
                # Use float64 for numerically stable color accumulation
                pixel_sum = np.sum(pixels_in_voxel, axis=0, dtype=np.float64)
                
                if key not in class_voxels[cls_name]:
                    class_voxels[cls_name][key] = {
                        'sum': np.array([0., 0., 0.], dtype=np.float64),
                        'count': 0
                    }
                
                class_voxels[cls_name][key]['sum'] += pixel_sum
                class_voxels[cls_name][key]['count'] += count

    # 3. Final aggregation: compute centroids and weights, then save to YAML
    final_priors = {}
    
    print("\nCalculating Final Priors...")
    for cls_name, voxels in class_voxels.items():
        voxel_list = []
        total_pixels = sum(v['count'] for v in voxels.values())
        
        # Compute a global median color across all instances of this class.
        # This acts as a fallback or default color prior.
        medians_list = np.array(class_instance_medians[cls_name])
        global_median = (
            np.median(medians_list, axis=0).tolist()
            if len(medians_list) > 0 else
            [128, 128, 128]
        )
        
        # Build a list of voxel entries for this class
        for key, data in voxels.items():
            count = data['count']
            
            # Skip voxels with very low support:
            # - less than 50 pixels, OR
            # - less than 0.2% of all pixels for this class
            threshold = max(50, 0.002 * total_pixels)
            if count < threshold:
                continue
                
            centroid = (data['sum'] / count).tolist()
            weight = float(count / total_pixels)
            
            voxel_list.append({
                'id': key,          # e.g., "rIdx_gIdx_bIdx"
                'centroid': centroid,  # average RGB color in this voxel
                'weight': weight       # relative importance / frequency
            })
            
        final_priors[cls_name] = {
            'total_pixels': int(total_pixels),
            'global_median': global_median,  # default color prior for the class
            'voxels': voxel_list
        }

    # Write priors to a YAML file for later use
    with open(output_path, 'w') as f:
        yaml.dump({'classes': final_priors}, f, default_flow_style=None)
        
    print(f"Robust Priors saved to {output_path}")


if __name__ == "__main__":
    """
    Command-line entry point.

    High-level behavior:
    - Parse CLI arguments for:
        * --data  : one or more folders of training images
        * --out   : output YAML file path
        * --model : YOLO segmentation model path
    - Call process_dataset() to build and save color priors.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs='+', required=True,
                        help="Folder(s) with clean training images")
    parser.add_argument("--out", default="robust_priors.yaml",
                        help="Output YAML file for saved priors")
    parser.add_argument("--model", default="yolov8s-seg.pt",
                        help="YOLO segmentation model path")
    args = parser.parse_args()
    
    process_dataset(args.data, args.model, args.out)
