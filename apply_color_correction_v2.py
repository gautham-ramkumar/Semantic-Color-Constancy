"""
Author: Sai Vamsi Rithvik Allanka, Yoga Srinivas Reddy Kasireddy, Gautham Ramkumar
Date: 2025-12-11
CS7180 - Advanced Perception
"""



import os
import glob
import math
import argparse
import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO


NUM_BINS = 8
BIN_SIZE = 256 // NUM_BINS


# ============================================================
# Priors utilities
# ============================================================

def load_class_priors(priors_path):
    """
    Load class color priors from a YAML file and convert them into a
    convenient lookup format for inference.

    For each class:
      - Reads the voxel entries (id, centroid, weight).
      - Converts each voxel id ("r_g_b") into RGB value ranges.
      - Approximates pixel count in each voxel from weight * total_pixels.
      - Sorts the voxels by count so common colors are checked first.

    Returns:
        priors_by_class: dict mapping class_name -> {
            "total_pixels": int,
            "ranges": [
               {
                 "R": [R_min, R_max],
                 "G": [G_min, G_max],
                 "B": [B_min, B_max],
                 "count": approx_count,
                 "center": np.array([R_c, G_c, B_c], float32),
                 "weight": float
               },
               ...
            ],
            "global_median": [R, G, B] or None
        }
    """
    with open(priors_path, "r") as f:
        priors = yaml.safe_load(f)

    classes = priors.get("classes", {})
    priors_by_class = {}

    for cls_name, info in classes.items():
        total_pixels = int(info.get("total_pixels", 0))
        voxels = info.get("voxels", [])
        global_median = info.get("global_median", None)

        processed = []
        for v in voxels:
            voxel_id = v.get("id", "")
            centroid = np.array(v.get("centroid", [128, 128, 128]), dtype=np.float32)
            weight = float(v.get("weight", 0.0))

            # Reconstruct bin ranges from voxel index "r_g_b"
            try:
                r_idx, g_idx, b_idx = map(int, voxel_id.split("_"))
            except Exception:
                # If id is malformed, skip this voxel
                continue

            R_min = int(r_idx * BIN_SIZE)
            R_max = int((r_idx + 1) * BIN_SIZE - 1)
            G_min = int(g_idx * BIN_SIZE)
            G_max = int((g_idx + 1) * BIN_SIZE - 1)
            B_min = int(b_idx * BIN_SIZE)
            B_max = int((b_idx + 1) * BIN_SIZE - 1)

            # approximate pixel count from weight * total_pixels
            approx_count = int(round(weight * total_pixels))
            if approx_count <= 0:
                # very tiny voxels are not useful
                continue

            processed.append({
                "R": [R_min, R_max],
                "G": [G_min, G_max],
                "B": [B_min, B_max],
                "count": approx_count,
                "center": centroid,
                "weight": weight,
            })

        # Sort most important bins first (by count, like before)
        processed.sort(key=lambda d: d["count"], reverse=True)

        priors_by_class[cls_name] = {
            "total_pixels": total_pixels,
            "ranges": processed,
            "global_median": global_median,
        }

    return priors_by_class


def get_class_prior_ranges(priors_by_class, cls_name, top_k=None):
    """
    Return the list of color ranges (bins) for a given class, optionally
    truncated to the top-k most populated bins.
    """
    if cls_name not in priors_by_class:
        return []

    ranges = priors_by_class[cls_name]["ranges"]
    if top_k is not None and top_k > 0:
        return ranges[:top_k]
    return ranges


def color_in_priors(color, class_ranges, margin=5.0):
    """
    Check whether a corrected color lies inside any of the stored prior
    ranges for a class.

    Args:
        color: np.array([R, G, B]) corrected color estimate.
        class_ranges: list of prior bins with "R", "G", "B" intervals.
        margin: expands each interval on both sides by this amount.

    Returns:
        inside (bool): True if color fits at least one prior bin.
    """
    R, G, B = float(color[0]), float(color[1]), float(color[2])
    for r in class_ranges:
        R_min, R_max = r["R"]
        G_min, G_max = r["G"]
        B_min, B_max = r["B"]

        if (R_min - margin <= R <= R_max + margin and
            G_min - margin <= G <= G_max + margin and
            B_min - margin <= B <= B_max + margin):
            return True
    return False


# ============================================================
# Image + YOLO utilities
# ============================================================

def find_images(root_dir):
    """
    Recursively search a directory for common image file types and return
    a sorted list of matching paths.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(root_dir, "**", e), recursive=True))
    return sorted(paths)


def resize_mask_to_image(mask, img_h, img_w):
    """
    Resize a YOLO mask to match the given image resolution and return a
    boolean mask.
    """
    mask = mask.astype(np.float32)
    resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return resized > 0.5


def compute_grey_world_illuminant(img_rgb):
    """
    Estimate the scene illuminant using the Grey-World assumption.

    The method:
      - Computes the global mean RGB of the image.
      - Uses the per-channel mean divided by the overall mean as the gain
        for each channel.

    Returns:
        illum: np.array([R_gain, G_gain, B_gain]) in float32.
    """
    img = img_rgb.astype(np.float32)
    m = img.mean(axis=(0, 1))  # [mR, mG, mB]
    mu = float(m.mean())
    eps = 1e-6
    illum = (m + eps) / (mu + eps)
    return illum.astype(np.float32)


def apply_color_correction(img_rgb, illum):
    """
    Apply diagonal color correction using an estimated illuminant and rescale
    intensities to keep the output in a reasonable range.

    Steps:
      - Divide each channel by its illuminant gain.
      - Use the 99.9th percentile of the corrected intensities to rescale
        the image to the 0–255 range.
      - Clip and convert back to uint8.
    """
    img = img_rgb.astype(np.float32)
    corrected = img / (illum[None, None, :] + 1e-6)
    max_val = np.percentile(corrected, 99.9)  # use high percentile instead of max
    if max_val > 0:
        corrected = corrected * (255.0 / max_val)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return corrected


# ============================================================
# Object extraction
# ============================================================

def extract_objects_with_priors(img_rgb, model, priors_by_class, conf_threshold=0.5, min_pixels=200):
    """
    Run YOLO segmentation on an image and extract only those object masks
    for which class color priors are available.

    For each detection:
      - Resizes the segmentation mask to the image size.
      - Discards small masks (area < min_pixels).
      - Computes the mean RGB of the pixels covered by the mask.

    Returns:
        objects: list of dicts with:
          - "class_name": class label string
          - "mask_bool":  H x W boolean mask
          - "num_pixels": pixel count inside the mask
          - "obs_mean":   np.array([R, G, B]) mean color in the input image
    """
    H, W = img_rgb.shape[:2]
    results = model(img_rgb, conf=conf_threshold, verbose=False)

    if len(results) == 0:
        return []

    res = results[0]
    if res.masks is None or res.boxes is None or len(res.masks.data) == 0:
        return []

    masks = res.masks.data  # (N, h_mask, w_mask)
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)  # (N,)
    names = model.names

    objects = []

    for i in range(len(cls_ids)):
        cls_id = cls_ids[i]
        cls_name = names[int(cls_id)]

        if cls_name not in priors_by_class:
            continue  # we have no priors for this class

        mask_small = masks[i].cpu().numpy()
        mask_bool = resize_mask_to_image(mask_small, H, W)
        num_pixels = int(mask_bool.sum())
        if num_pixels < min_pixels:
            continue

        pixels = img_rgb[mask_bool]  # (N, 3)
        if pixels.size == 0:
            continue

        obs_mean = pixels.mean(axis=0).astype(np.float32)  # [R,G,B]

        objects.append({
            "class_name": cls_name,
            "mask_bool": mask_bool,
            "num_pixels": num_pixels,
            "obs_mean": obs_mean,
        })

    return objects


# ============================================================
# Hypothesis search & scoring
# ============================================================

def stabilize_illum(illum, min_ratio=0.5, max_ratio=2.0):
    """
    Clamp the relative channel ratios of an illuminant estimate so that
    extreme values are avoided.

    The vector is first normalized to mean 1, then each channel is clipped
    to [min_ratio, max_ratio], and finally renormalized to mean 1 again.
    """
    illum = np.asarray(illum, dtype=np.float32)
    eps = 1e-6

    # First normalize to mean 1
    illum = illum / (illum.mean() + eps)

    # Clamp each channel
    illum = np.clip(illum, min_ratio, max_ratio)

    # Re-normalize to mean 1
    illum = illum / (illum.mean() + eps)

    return illum.astype(np.float32)


def blend_with_grey_world(illum_semantic, img_rgb, weight=0.5):
    """
    Blend a prior-based (semantic) illuminant estimate with the Grey-World
    estimate from the image.

    L_final = (1 - weight) * L_GW + weight * L_semantic
    """
    illum_semantic = np.asarray(illum_semantic, dtype=np.float32)
    L_gw = compute_grey_world_illuminant(img_rgb)
    w = float(weight)
    L_blend = (1.0 - w) * L_gw + w * illum_semantic
    return L_blend.astype(np.float32)


def compute_candidate_illuminants_from_anchor(anchor_obj, anchor_ranges, eps=1e-6):
    """
    Generate candidate illuminants from a single anchor object and its
    class priors.

    For each prior bin center c in anchor_ranges:
      L_candidate = obs_mean / c,
    where obs_mean is the observed mean RGB of the anchor object.

    Very dark or very bright bins are skipped because they are less reliable
    as anchors.

    Returns:
        candidates: list of np.array([R_gain, G_gain, B_gain])
    """
    obs_mean = anchor_obj["obs_mean"]  # [R,G,B]
    candidates = []

    for r in anchor_ranges:
        center = r["center"]  # [R_c, G_c, B_c]

        # Skip very dark / very bright bins as anchors
        brightness = float(center.mean())
        if brightness < 40.0 or brightness > 220.0:
            continue

        # Avoid division by zero or extremely small values
        denom = np.maximum(center, eps)
        L = (obs_mean + eps) / denom
        candidates.append(L.astype(np.float32))

    return candidates


def score_illuminant_with_other_objects(
    illum,
    objects,
    priors_by_class,
    anchor_index,
    margin=5.0
):
    """
    Compute how well a candidate illuminant explains all non-anchor objects
    in the scene.

    For each non-anchor object:
      - Correct its observed mean using 'illum'.
      - Check if the corrected color lies inside the priors for that class.
      - If it does, add the object's pixel count to the score.

    Returns:
        score (float): total support in pixels for this illuminant.
    """
    score = 0.0

    for idx, obj in enumerate(objects):
        if idx == anchor_index:
            continue  # skip the anchor itself

        cls_name = obj["class_name"]
        obs_mean = obj["obs_mean"]

        # Correct this object's mean color under this illuminant
        eps = 1e-6
        corrected_mean = obs_mean / (illum + eps)  # [R_corr, G_corr, B_corr]

        class_ranges = get_class_prior_ranges(priors_by_class, cls_name)
        inside = color_in_priors(corrected_mean, class_ranges, margin=margin)

        if inside:
            # weight by number of pixels for this object
            score += float(obj["num_pixels"])

    return score


def choose_illuminant_multi_object(
    img_rgb,
    objects,
    priors_by_class,
    top_k_anchor_bins=5,
    margin=5.0
):
    """
    Estimate the illuminant when there are two or more objects with priors.

    Steps:
      1. Select the largest object as the anchor.
      2. Build candidate illuminants from the top-k bins for the anchor.
      3. For each candidate, measure how many pixels from the other objects
         become consistent with their priors.
      4. Add a penalty if the candidate deviates too much from Grey-World
         in log space, and choose the best-scoring candidate.
    """
    # Choose anchor as object with largest area
    anchor_index = int(np.argmax([o["num_pixels"] for o in objects]))
    anchor = objects[anchor_index]
    anchor_cls = anchor["class_name"]

    anchor_ranges = get_class_prior_ranges(
        priors_by_class,
        anchor_cls,
        top_k=top_k_anchor_bins
    )
    if not anchor_ranges:
        # No priors for anchor class (shouldn't happen if we filtered earlier)
        return compute_grey_world_illuminant(img_rgb)

    # If no valid candidates, fall back
    candidate_illums = compute_candidate_illuminants_from_anchor(anchor, anchor_ranges)
    if not candidate_illums:
        return compute_grey_world_illuminant(img_rgb)

    # Grey-World regularization setup
    L_gw = compute_grey_world_illuminant(img_rgb)
    total_pixels = sum(o["num_pixels"] for o in objects)
    # Regularization strength: tune this (0.001 is a reasonable starting point)
    lambda_reg = 0.001 * float(total_pixels)

    best_score = -1e18
    best_illum = None

    for illum in candidate_illums:
        # 1) object-fit score
        score_objects = score_illuminant_with_other_objects(
            illum,
            objects,
            priors_by_class,
            anchor_index=anchor_index,
            margin=margin
        )

        # 2) penalty if illum is far from Grey-World in log space
        eps = 1e-6
        dist_log = np.linalg.norm(
            np.log(illum + eps) - np.log(L_gw + eps),
            ord=2
        )
        penalty = lambda_reg * dist_log

        score_total = score_objects - penalty

        if score_total > best_score:
            best_score = score_total
            best_illum = illum

    if best_illum is None:
        best_illum = compute_grey_world_illuminant(img_rgb)

    return best_illum.astype(np.float32)


def choose_illuminant_single_object(
    img_rgb,
    obj,
    priors_by_class,
    grey_world_weight=True,
    top_k_bins=10,
):
    """
    Estimate the illuminant when exactly one object with priors is present.

    Steps:
      - Build candidate illuminants from the class priors for that object.
      - If grey_world_weight is True, pick the candidate closest to
        Grey-World in log space; otherwise, take the first candidate.
    """
    cls_name = obj["class_name"]
    class_ranges = get_class_prior_ranges(
        priors_by_class,
        cls_name,
        top_k=top_k_bins
    )
    if not class_ranges:
        return compute_grey_world_illuminant(img_rgb)

    candidate_illums = compute_candidate_illuminants_from_anchor(
        obj,
        class_ranges
    )
    if not candidate_illums:
        return compute_grey_world_illuminant(img_rgb)

    L_gw = compute_grey_world_illuminant(img_rgb)
    if not grey_world_weight:
        # If we don't want to use Grey-World, just pick the first candidate
        return candidate_illums[0].astype(np.float32)

    best_score = -1e18
    best_illum = None
    eps = 1e-6

    for L in candidate_illums:
        # Use negative distance in log space as score
        d = np.linalg.norm(
            np.log(L + eps) - np.log(L_gw + eps),
            ord=2
        )
        s = -d  # smaller distance = higher score
        if s > best_score:
            best_score = s
            best_illum = L

    if best_illum is None:
        best_illum = L_gw

    return best_illum.astype(np.float32)


def print_estimation_summary(objects, mode, illum):
    """
    Print a short summary of how the illuminant was estimated for a single
    image: number of objects used, class counts, mode, and final gains.
    """
    M = len(objects)

    # Count how many of each class we have
    class_counts = {}
    for obj in objects:
        cls = obj["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print("--------------------------------------------------")
    print(f"[INFO] Objects with priors in this image: {M}")
    if M > 0:
        print(f"[INFO] Class counts: {class_counts}")

    # Explain the mode
    if mode == "grey_world_only":
        print("[INFO] Mode: grey_world_only (no usable objects with priors, using Grey-World fallback).")
    elif mode.startswith("single_object_"):
        anchor_cls = mode.replace("single_object_", "")
        print(f"[INFO] Mode: single_object (only one object with priors: '{anchor_cls}').")
        print("[INFO]         Used the object's priors + Grey-World to choose the illuminant.")
    elif mode.startswith("multi_object_"):
        print(f"[INFO] Mode: multi_object (hypothesis scoring with {M} objects).")
        print("[INFO]         Used one anchor object to generate candidate tints;")
        print("[INFO]         scored each tint by how well other objects fit their priors.")
    else:
        print(f"[INFO] Mode: {mode}")

    print(f"[INFO] Estimated illuminant (R, G, B gains): {illum}")
    print("--------------------------------------------------")


def estimate_illuminant(
    img_rgb,
    model,
    priors_by_class,
    conf_threshold=0.5,
    min_pixels=200,
    top_k_anchor_bins=5,
    top_k_single_bins=10,
    margin=5.0,
):
    """
    Decide which strategy to use to estimate the illuminant for a given
    image and return the final stabilized estimate.

    Branches:
      - 0 objects with priors: Grey-World only.
      - 1 object with priors: single-object strategy + Grey-World blend.
      - 2+ objects with priors: multi-object hypothesis scoring + blend.
    """
    objects = extract_objects_with_priors(
        img_rgb,
        model,
        priors_by_class,
        conf_threshold=conf_threshold,
        min_pixels=min_pixels,
    )

    M = len(objects)

    if M == 0:
        # Pure Grey-World fallback
        illum = compute_grey_world_illuminant(img_rgb)
        mode = "grey_world_only"
    elif M == 1:
        # Semantic estimate from single object
        illum_sem = choose_illuminant_single_object(
            img_rgb,
            objects[0],
            priors_by_class,
            grey_world_weight=True,
            top_k_bins=top_k_single_bins,
        )
        mode = f"single_object_{objects[0]['class_name']}"

        # Blend semantic with Grey-World (e.g. 50/50)
        illum = blend_with_grey_world(illum_sem, img_rgb, weight=0.5)

    else:
        # Semantic estimate from multi-object hypothesis scoring
        illum_sem = choose_illuminant_multi_object(
            img_rgb,
            objects,
            priors_by_class,
            top_k_anchor_bins=top_k_anchor_bins,
            margin=margin
        )
        mode = f"multi_object_{M}"

        # Blend semantic with Grey-World (e.g. 50/50)
        illum = blend_with_grey_world(illum_sem, img_rgb, weight=0.5)

    # Finally, stabilize to avoid crazy channel ratios
    illum = stabilize_illum(illum, min_ratio=0.3, max_ratio=5.0)

    print_estimation_summary(objects, mode, illum)

    return illum.astype(np.float32), mode


# ============================================================
# Main pipeline: single image or folder
# ============================================================

def process_image(
    img_path,
    model,
    priors_by_class,
    conf_threshold=0.5,
    min_pixels=200,
    top_k_anchor_bins=5,
    top_k_single_bins=10,
    margin=5.0,
    save_dir=None,
):
    """
    Run the full color-correction pipeline on a single image:
      - load the image,
      - estimate the illuminant,
      - apply color correction,
      - save the corrected result.
    """
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Warning: cannot read {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    illum, mode = estimate_illuminant(
        img_rgb,
        model=model,
        priors_by_class=priors_by_class,
        conf_threshold=conf_threshold,
        min_pixels=min_pixels,
        top_k_anchor_bins=top_k_anchor_bins,
        top_k_single_bins=top_k_single_bins,
        margin=margin,
    )

    corrected_rgb = apply_color_correction(img_rgb, illum)
    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)

    if save_dir is None:
        save_dir = os.path.dirname(img_path)
    os.makedirs(save_dir, exist_ok=True)

    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)

    if save_dir is None:
        # same folder as input: keep _corrected
        save_dir = os.path.dirname(img_path)
        filename = f"{name}_corrected{ext}"
    else:
        # custom out_dir: no suffix
        filename = f"{name}{ext}"

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)


    cv2.imwrite(out_path, corrected_bgr)
    print(f"{img_path} -> {out_path}, mode={mode}, illum={illum}")


def main():
    """
    Parse command-line arguments, load priors and YOLO model, and run the
    color-correction pipeline on either a single image or all images in a
    folder.
    """
    parser = argparse.ArgumentParser(
        description="Object-aware color correction using full class priors."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=str,
        help="Path to a single image to correct.",
    )
    group.add_argument(
        "--data",
        type=str,
        help="Folder with images (processed recursively).",
    )
    parser.add_argument(
        "--priors",
        type=str,
        required=True,
        help="Path to color_priors.yaml produced by build_color_priors.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="YOLO segmentation model path/name.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=200,
        help="Minimum pixels in a mask to consider an object.",
    )
    parser.add_argument(
        "--top_k_anchor_bins",
        type=int,
        default=5,
        help="Number of top prior bins to use for anchor object hypotheses.",
    )
    parser.add_argument(
        "--top_k_single_bins",
        type=int,
        default=10,
        help="Number of top prior bins to use in single-object case.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=5.0,
        help="RGB margin when testing if corrected color is inside priors.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional output directory. "
             "If empty, corrected images are saved next to originals.",
    )

    args = parser.parse_args()

    print(f"Loading priors from: {args.priors}")
    priors_by_class = load_class_priors(args.priors)
    print("Loaded priors for classes:", list(priors_by_class.keys()))

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    if args.image:
        process_image(
            args.image,
            model=model,
            priors_by_class=priors_by_class,
            conf_threshold=args.conf,
            min_pixels=args.min_pixels,
            top_k_anchor_bins=args.top_k_anchor_bins,
            top_k_single_bins=args.top_k_single_bins,
            margin=args.margin,
            save_dir=args.out_dir if args.out_dir else None,
        )
    else:
        image_paths = find_images(args.data)
        print(f"Found {len(image_paths)} images in {args.data}")
        for p in tqdm(image_paths, desc="Correcting images"):
            process_image(
                p,
                model=model,
                priors_by_class=priors_by_class,
                conf_threshold=args.conf,
                min_pixels=args.min_pixels,
                top_k_anchor_bins=args.top_k_anchor_bins,
                top_k_single_bins=args.top_k_single_bins,
                margin=args.margin,
                save_dir=args.out_dir if args.out_dir else None,
            )


if __name__ == "__main__":
    main()
