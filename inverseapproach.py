"""
Inverse semantic color constancy correction using YOLO segmentation and color priors.
Author: Gautham Ramkumar, Yoga Srinivas Reddy Kasireddy, Sai Vamsi Rithvik Allanka
Date: 2024-06-15
Extension - 1: Inverse Semantic Approach for Color Constancy for single object Domainant Images
CS7180 - Advanced Perception
"""


import os
import argparse
import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_color_priors(yaml_path):
    """Load semantic color priors from YAML file"""
    with open(yaml_path, 'r') as f:
        priors = yaml.safe_load(f)
    return priors


def debug_priors_structure(priors, class_name):
    """Print the structure of a class in priors for debugging"""
    if 'classes' not in priors or class_name not in priors['classes']:
        print(f"  Class '{class_name}' not found in priors")
        return
    
    class_data = priors['classes'][class_name]
    print(f"  Structure of '{class_name}':")
    print(f"  Keys: {class_data.keys()}")
    for key in class_data.keys():
        val = class_data[key]
        if isinstance(val, list):
            print(f"  '{key}': list of {len(val)} items")
            if len(val) > 0:
                print(f"    First item: {val[0]}")
        elif isinstance(val, dict):
            print(f"  '{key}': dict with keys {val.keys()}")
        else:
            print(f"  '{key}': {type(val).__name__} = {val}")


def get_prior_colors(priors, class_name, verbose=False):
    """Extract prior RGB colors for a class, sorted by count"""
    if 'classes' not in priors or class_name not in priors['classes']:
        if verbose:
            print(f"  Class '{class_name}' not in priors")
        return []
    
    class_data = priors['classes'][class_name]
    total_pixels = float(class_data.get('total_pixels', 1))
    
    prior_colors = []
    
    if 'ranges' in class_data:
        ranges = class_data['ranges']
        if verbose:
            print(f"  Found {len(ranges)} ranges")
        
        for r in ranges:
            r_bin = r.get('R', [0, 255])
            g_bin = r.get('G', [0, 255])
            b_bin = r.get('B', [0, 255])
            count = int(r.get('count', 0))
            
            centroid = np.array([
                (r_bin[0] + r_bin[1]) / 2.0,
                (g_bin[0] + g_bin[1]) / 2.0,
                (b_bin[0] + b_bin[1]) / 2.0
            ], dtype=np.float32)
            
            weight = float(count) / total_pixels if total_pixels > 0 else 0.0
            
            prior_colors.append({
                'rgb': centroid,
                'weight': weight,
                'count': count
            })
    else:
        if verbose:
            print(f"  No 'ranges' key found")
        return []
    
    prior_colors.sort(key=lambda x: x['count'], reverse=True)
    if verbose:
        print(f"  Extracted {len(prior_colors)} prior colors")
        for i, c in enumerate(prior_colors[:3]):
            print(f"  Prior {i}: RGB={c['rgb']}, count={c['count']}")
    return prior_colors


def extract_objects(img_rgb, model, priors, conf_threshold=0.5, min_pixels=100, verbose=False):
    """Detect objects with YOLO and extract those with known priors"""
    H, W = img_rgb.shape[:2]
    results = model(img_rgb, conf=conf_threshold, verbose=False)
    
    if not results or results[0].masks is None:
        if verbose:
            print("  YOLO detected no objects or no masks")
        return []
    
    res = results[0]
    masks = res.masks.data
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    names = model.names
    
    prior_classes = set(priors.get('classes', {}).keys())
    if verbose:
        print(f"  Prior classes available: {prior_classes}")
    
    detected_classes = [names[int(cid)] for cid in cls_ids]
    if verbose:
        print(f"  YOLO detected classes: {detected_classes}")
    
    objects = []
    
    for i in range(len(cls_ids)):
        cls_id = int(cls_ids[i])
        cls_name = names[cls_id]
        
        if 'classes' not in priors or cls_name not in priors['classes']:
            if verbose:
                print(f"  Skipping '{cls_name}' - no priors")
            continue
        
        mask_small = masks[i].cpu().numpy()
        mask_resized = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized > 0.5
        
        num_pixels = int(mask_bool.sum())
        if num_pixels < min_pixels:
            if verbose:
                print(f"  '{cls_name}' has {num_pixels} pixels (need {min_pixels})")
            continue
        
        pixels = img_rgb[mask_bool].astype(np.float32)
        obs_mean = np.mean(pixels, axis=0)
        
        objects.append({
            'class_name': cls_name,
            'obs_mean': obs_mean,
            'num_pixels': num_pixels,
            'mask': mask_bool
        })
    
    if verbose:
        print(f"  Extracted {len(objects)} valid objects")
    
    return objects


def normalize_illuminant(illum, eps=1e-6):
    """Normalize illuminant so mean(illum) = 1.0 and clamp extreme values"""
    mean_val = np.mean(illum) + eps
    normalized = illum / mean_val
    clipped = np.clip(normalized, 0.1, 10.0)
    final = clipped / np.mean(clipped)
    return final.astype(np.float32)


def estimate_illuminant_from_object(obs_color, prior_color, eps=1e-6):
    """Invert observed color using prior: L = obs_color / prior_color"""
    denom = np.maximum(prior_color, eps)
    L = (obs_color + eps) / denom
    return L.astype(np.float32)


def score_illuminant_multiobject(illum, objects, priors, anchor_idx, eps=1e-6):
    """Score illuminant by how many other objects fit their priors after correction"""
    score = 0.0
    
    for idx, obj in enumerate(objects):
        if idx == anchor_idx:
            continue
        
        cls_name = obj['class_name']
        obs_mean = obj['obs_mean']
        prior_colors = get_prior_colors(priors, cls_name)
        
        if not prior_colors:
            continue
        
        corrected_mean = obs_mean / (illum + eps)
        best_dist = float('inf')
        
        for prior_info in prior_colors[:5]:
            prior_rgb = prior_info['rgb']
            dist = np.linalg.norm(corrected_mean - prior_rgb)
            best_dist = min(best_dist, dist)
        
        if best_dist < 50:
            score += float(obj['num_pixels']) / best_dist
    
    return score


def choose_illuminant_single_object(obj, priors, verbose=False):
    """Single object: generate candidates from priors, pick most common"""
    cls_name = obj['class_name']
    prior_colors = get_prior_colors(priors, cls_name, verbose)
    
    if not prior_colors:
        if verbose:
            print(f"[WARN] No prior colors for {cls_name}")
        return None
    
    best_illum = None
    best_score = -1e18
    
    for prior_info in prior_colors:
        prior_rgb = prior_info['rgb']
        weight = prior_info['weight']
        
        L_cand = estimate_illuminant_from_object(obj['obs_mean'], prior_rgb)
        L_norm = normalize_illuminant(L_cand)
        
        if verbose:
            print(f"  Prior RGB={prior_rgb}, Weight={weight:.6f}, Illum={L_norm}")
        
        score = weight
        
        if score > best_score:
            best_score = score
            best_illum = L_norm
    
    if verbose:
        print(f"  Selected illuminant: {best_illum}")
    return best_illum


def choose_illuminant_multi_object(objects, priors, verbose=False):
    """Multi-object: use largest object as anchor, score candidates with others"""
    anchor_idx = np.argmax([o['num_pixels'] for o in objects])
    anchor = objects[anchor_idx]
    
    cls_name = anchor['class_name']
    prior_colors = get_prior_colors(priors, cls_name, verbose)
    
    if not prior_colors:
        if verbose:
            print(f"[WARN] No prior colors for anchor {cls_name}")
        return None
    
    best_illum = None
    best_score = -1e18
    
    for prior_info in prior_colors:
        prior_rgb = prior_info['rgb']
        
        L_cand = estimate_illuminant_from_object(anchor['obs_mean'], prior_rgb)
        L_norm = normalize_illuminant(L_cand)
        
        score = score_illuminant_multiobject(L_norm, objects, priors, anchor_idx)
        
        if verbose:
            print(f"  Prior RGB={prior_rgb}, Score={score:.2f}, Illum={L_norm}")
        
        if score > best_score:
            best_score = score
            best_illum = L_norm
    
    if verbose:
        print(f"  Selected illuminant: {best_illum} with score {best_score:.2f}")
    return best_illum


def estimate_illuminant(img_rgb, model, priors, conf_threshold=0.5, min_pixels=100, verbose=False):
    """Main dispatch: extract objects and choose illuminant based on count"""
    objects = extract_objects(img_rgb, model, priors, conf_threshold, min_pixels, verbose)
    
    if verbose:
        print(f"Detected {len(objects)} objects with priors: {[o['class_name'] for o in objects]}")
    
    if len(objects) == 0:
        return None
    elif len(objects) == 1:
        illum = choose_illuminant_single_object(objects[0], priors, verbose)
    else:
        illum = choose_illuminant_multi_object(objects, priors, verbose)
    
    return illum


def apply_color_correction(img_rgb, illum):
    """Apply illuminant correction: img_corrected = img / illum"""
    img_float = img_rgb.astype(np.float32)
    corrected = img_float / illum[np.newaxis, np.newaxis, :]
    
    max_val = np.percentile(corrected, 99.5)
    if max_val > 0:
        corrected = corrected * (255.0 / max_val)
    
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def process_image(img_path, model, priors, conf_threshold=0.5, min_pixels=100, out_dir=None, verbose=False):
    """Load image, estimate illuminant, apply correction, save output"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot read {img_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    illum = estimate_illuminant(img_rgb, model, priors, conf_threshold, min_pixels, verbose)
    
    if illum is None:
        print(f"[SKIP] {img_path} - could not estimate illuminant")
        return
    
    if verbose:
        print(f"Estimated illuminant: {illum}")
    
    corrected_rgb = apply_color_correction(img_rgb, illum)
    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
    
    if out_dir is None or out_dir == '':
        out_dir = os.path.dirname(img_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}_inverse_corrected{ext}")
    
    cv2.imwrite(out_path, corrected_bgr)
    print(f"  {img_path} -> {out_path}")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Inverse semantic color constancy correction"
    )
    
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--priors', type=str, required=True, help='Path to color_priors.yaml')
    parser.add_argument('--model', type=str, default='yolov8s-seg.pt', help='YOLO segmentation model')
    parser.add_argument('--conf', type=float, default=0.5, help='YOLO confidence threshold')
    parser.add_argument('--min_pixels', type=int, default=100, help='Minimum pixels in mask')
    parser.add_argument('--out_dir', type=str, default='', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"Loading priors: {args.priors}")
    priors = load_color_priors(args.priors)
    print(f"Classes in priors: {list(priors.get('classes', {}).keys())}")
    
    if args.verbose:
        debug_priors_structure(priors, 'microwave')
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    process_image(args.image, model, priors, args.conf, args.min_pixels, args.out_dir, args.verbose)


if __name__ == '__main__':
    main()