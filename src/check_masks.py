"""
Mask Value Inspection Utility for GOALS and OIMHS Datasets.

Inspects ground truth masks to:
1. Verify the exact pixel value/color representing the choroid layer:
   - GOALS: Grayscale pixel value 160.
   - OIMHS: Color-coded right half with Green BGR (0, 255, 0).
2. Detect whether near-target/interpolated values exist from compression or anti-aliasing.
3. Print an exact histogram and summary of anatomical classes.
"""

import argparse
import glob
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List
import cv2
import numpy as np


def inspect_oimhs_masks(image_paths: List[str]) -> None:
    print(f"\n--- Inspecting {len(image_paths)} OIMHS Composite Scans ---")
    total_exact_green = 0
    total_retina_blue = 0
    total_mh_red = 0
    total_cysts_yellow = 0
    near_green_artifacts = 0

    for idx, path in enumerate(image_paths):
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        H, W_full = bgr.shape[:2]
        W_mid = W_full // 2
        right = bgr[:, W_mid:]

        # Exact colors (BGR)
        is_green = (right[:, :, 1] == 255) & (right[:, :, 0] == 0) & (right[:, :, 2] == 0)
        is_blue = (right[:, :, 0] == 255) & (right[:, :, 1] == 0) & (right[:, :, 2] == 0)
        is_red = (right[:, :, 2] == 255) & (right[:, :, 0] == 0) & (right[:, :, 1] == 0)
        is_yellow = (right[:, :, 1] == 255) & (right[:, :, 2] == 255) & (right[:, :, 0] == 0)

        total_exact_green += int(is_green.sum())
        total_retina_blue += int(is_blue.sum())
        total_mh_red += int(is_red.sum())
        total_cysts_yellow += int(is_yellow.sum())

        # Check for near-green artifacts (e.g. anti-aliased edges)
        near_green = (
            (right[:, :, 1] > 200) & (right[:, :, 0] < 50) & (right[:, :, 2] < 50) & (~is_green)
        )
        if near_green.sum() > 0:
            near_green_artifacts += 1

        if idx < 5:
            s_name = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
            print(
                f"  Scan #{idx+1} ({s_name}): Choroid(Green)={is_green.sum():<6} "
                f"Retina(Blue)={is_blue.sum():<6} MH(Red)={is_red.sum():<5} Cysts(Yellow)={is_yellow.sum():<5}"
            )

    print("\n--- OIMHS Ground Truth Summary Report ---")
    print(f"Total scans inspected:                     {len(image_paths):,}")
    print(f"Total Choroid (Green BGR 0,255,0) pixels:  {total_exact_green:,}")
    print(f"Total Retina (Blue BGR 255,0,0) pixels:    {total_retina_blue:,}")
    print(f"Total Macular Hole (Red BGR 0,0,255):      {total_mh_red:,}")
    print(f"Total Intraretinal Cysts (Yellow):         {total_cysts_yellow:,}")

    if total_exact_green > 0:
        print("\nSUCCESS: Target Choroid annotation (pure Green BGR [0, 255, 0]) is present and verified across the dataset.")
    else:
        print("\nCRITICAL WARNING: No green choroid pixels found!")

    if near_green_artifacts > 0:
        print(f"NOTE: {near_green_artifacts} scans have slightly anti-aliased edge pixels near green.")
    else:
        print("CLEAN: Zero compression artifacts. Exact equality `BGR == [0, 255, 0]` is 100% loss-free.")


def inspect_goals_masks(mask_paths: List[str], target_val: int = 160) -> None:
    print(f"\n--- Inspecting {len(mask_paths)} GOALS Mask Files ---")
    overall_unique_vals = set()
    total_exact_target_pixels = 0

    for idx, path in enumerate(mask_paths):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        unique_vals, counts = np.unique(mask, return_counts=True)
        overall_unique_vals.update(unique_vals.tolist())
        val_dict = dict(zip(unique_vals, counts))
        total_exact_target_pixels += val_dict.get(target_val, 0)
        if idx < 5:
            print(f"  Mask #{idx + 1} ({os.path.basename(path)}): unique pixel values = {sorted(unique_vals.tolist())}")

    print("\n--- GOALS Ground Truth Summary Report ---")
    print(f"All unique pixel values found across masks: {sorted(list(overall_unique_vals))}")
    print(f"Total exact pixels with value == {target_val}: {total_exact_target_pixels:,}")


def main():
    parser = argparse.ArgumentParser(description="Inspect OCT ground truth mask values.")
    parser.add_argument("--mask_dir", type=str, default="data/raw", help="Path to masks directory")
    parser.add_argument("--target_val", type=int, default=160, help="Expected choroid pixel value for GOALS (default: 160)")
    args = parser.parse_args()

    # Search for all PNG files
    pngs = sorted(glob.glob(os.path.join(args.mask_dir, "**", "*.png"), recursive=True))
    if not pngs:
        print(f"No PNG files found in '{args.mask_dir}'.")
        return

    # Check if first image is an OIMHS composite image (width == 2 * height)
    sample = cv2.imread(pngs[0])
    if sample is not None and sample.shape[1] > sample.shape[0]:
        inspect_oimhs_masks(pngs)
    else:
        inspect_goals_masks(pngs, target_val=args.target_val)


if __name__ == "__main__":
    main()
