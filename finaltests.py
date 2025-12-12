"""
Author: Gautham Ramkumar, Yoga Srinivas Reddy Kasireddy, Sai Vamsi Rithvik Allanka
Date: 2025-12-11
CS7180 - Advanced Perception
"""

import argparse
import os
import glob
import cv2
import numpy as np
import json
import math
from tqdm import tqdm
from ultralytics import YOLO
import yaml

## Function to calculate MSE
def calculate_mse(img1, img2):
    """
    PSNR is calculated as 20 * log10(max_pixel / sqrt(MSE)). Even though the the problem we address here is about color correction, 
    the PSNR formula is used to measure the similarity between the original and corrected images. For the purpose of this task, 
    we need to calculate MSE to use it in the PSNR calculation.
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return float(np.mean((img1 - img2) ** 2))

## Function to calculate PSNR
def calculate_psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

## Function to calculate Angular Error between two images
def calculate_angular_error(img1, img2):
    """
    Calculates the angular error (in degrees) between two images.
    This is illumination-invariant (ignores brightness differences). 
    The angular error is calculated as the average angle between corresponding pixels in the two images.
    We take the three RGB channels of the images as a vector and calculate the angle between these vectors.
    """
    # Reshape to list of pixels (N, 3)
    v1 = img1.reshape(-1, 3).astype(np.float32)
    v2 = img2.reshape(-1, 3).astype(np.float32)

    # Normalize vectors to unit length
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    
    # Avoid division by zero
    v1_u = v1 / (norm1 + 1e-6)
    v2_u = v2 / (norm2 + 1e-6)

    # Calculate Dot Product
    dot = np.sum(v1_u * v2_u, axis=1)
    
    # Clip for safety (acos requires -1 to 1)
    dot = np.clip(dot, -1.0, 1.0)

    # Calculate Angle
    angles = np.degrees(np.arccos(dot))
    
    return float(np.mean(angles))

## Validation function to evaluate the dataset
def evaluate_dataset(corrected_dir, gt_dir):
    # Find images in corrected directory
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    corrected_paths = []
    for e in exts:
        corrected_paths.extend(glob.glob(os.path.join(corrected_dir, "**", e), recursive=True))
    
    corrected_paths = sorted(corrected_paths)
    results = []

    print(f"Evaluating {len(corrected_paths)} images...")

    for c_path in tqdm(corrected_paths):
        fname = os.path.basename(c_path)
        
        # Determine the likely GT filename
        # Remove '_corrected' suffix if present to find the original file
        gt_fname = fname.replace("_corrected", "")
        
        # Try to find matching GT image
        gt_path = os.path.join(gt_dir, gt_fname)
        
        if not os.path.exists(gt_path):
            # Fallback 1: Try exact filename (maybe no suffix was added)
            gt_path = os.path.join(gt_dir, fname)
            
            if not os.path.exists(gt_path):
                # Fallback 2: Try searching recursively/ignoring extension
                found = False
                base_name = os.path.splitext(gt_fname)[0] # e.g. "img1" from "img1.jpg"
                
                for e in exts:
                    # Look for "img1.jpg", "img1.png", etc.
                    candidates = glob.glob(os.path.join(gt_dir, "**", f"{base_name}{e[1:]}"), recursive=True)
                    if not candidates:
                         # Try finding with original extension but recursive
                         candidates = glob.glob(os.path.join(gt_dir, "**", gt_fname), recursive=True)
                    
                    if candidates:
                        gt_path = candidates[0]
                        found = True
                        break
                if not found:
                    print(f"Warning: No GT found for {fname} (looked for {gt_fname})")
                    continue

        # Load Images
        img_c = cv2.imread(c_path)
        img_gt = cv2.imread(gt_path)

        if img_c is None or img_gt is None:
            print(f"Warning: Could not read images for {fname}")
            continue

        # Ensure shapes match
        if img_c.shape != img_gt.shape:
            img_c = cv2.resize(img_c, (img_gt.shape[1], img_gt.shape[0]))

        # Calculate Metrics (Only PSNR and Ang Err)
        psnr = calculate_psnr(img_c, img_gt)
        ang_err = calculate_angular_error(img_c, img_gt)

        results.append({
            "filename": fname,
            "psnr": psnr,
            "angular_error": ang_err
        })

    # Print Per-Image Metrics Table
    if results:
        print("\n" + "="*65)
        print(f"{'Filename':<35} | {'PSNR (dB)':<10} | {'Ang Err (deg)':<15}")
        print("-" * 65)
        for r in results:
            print(f"{r['filename']:<35} | {r['psnr']:<10.4f} | {r['angular_error']:<15.4f}")
        print("="*65)

    # Summary
    # Summary
    if results:
        psnrs = [r['psnr'] for r in results]
        ang_errors = [r['angular_error'] for r in results]

        avg_psnr = float(np.mean(psnrs))
        avg_ang = float(np.mean(ang_errors))          # Mean Angular Error (MAE)
        median_ang = float(np.median(ang_errors))     # Median Angular Error (MedAE)

        # Trimmed Mean Angular Error (TrimMean25): drop best 25% and worst 25%
        sorted_ang = np.sort(np.array(ang_errors))
        n = len(sorted_ang)
        trim_frac = 0.25
        k = int(n * trim_frac)

        if n > 2 * k:
            trimmed = sorted_ang[k:n - k]
            trim_mean_ang = float(np.mean(trimmed))
        else:
            # Not enough samples to trim properly; fall back to mean
            trim_mean_ang = avg_ang

        print("\n" + "="*60)
        print("SUMMARY METRICS")
        print("="*60)
        print(f"Total images evaluated      : {len(results)}")
        print(f"Average PSNR                : {avg_psnr:.4f} dB")
        print(f"Mean Angular Error (MAE)    : {avg_ang:.4f} degrees")
        print(f"Median Angular Error (MedAE): {median_ang:.4f} degrees")
        print(f"TrimMean25 Angular Error    : {trim_mean_ang:.4f} degrees")
        print("="*60)

        # Save detailed results
        with open("evaluation_metrics.json", "w") as f:
            json.dump(results, f, indent=4)

    else:
        print("No matching ground truth images found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Folder with CORRECTED images")
    parser.add_argument("--gt_data", required=True, help="Folder with GROUND TRUTH images")
    # Ignored args to prevent crashes if user reuses command line
    parser.add_argument("--priors", help="Ignored")
    parser.add_argument("--out_dir", help="Ignored")
    parser.add_argument("--model", help="Ignored")
    
    args = parser.parse_args()
    
    evaluate_dataset(args.data, args.gt_data)


    """
    Command to run the script:
    python3 finaltests.py --data (Corrected images folder path) --priors (YAML file path) --gt_data (Ground truth images folder path) 
    """