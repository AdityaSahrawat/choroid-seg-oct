"""
Dataset Splitting Script for GOALS OCT Challenge.

Ensures data integrity by splitting strictly by eye/subject whenever eye identifiers
are present in metadata or filenames.

Features:
- Scans for metadata CSV/Excel files if available.
- Prints exact count of total images vs. unique eye IDs.
- Warns prominently if total images == unique eye IDs (indicating filename-only numbering).
- Supports standard train/val/test splits AND 5-fold cross validation.

Outputs:
- data/splits/train.txt, val.txt, test.txt
- Optional K-fold directories: data/splits/fold_0/, fold_1/, ...
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def load_metadata_mapping(raw_dir: str, metadata_path: Optional[str] = None) -> Dict[str, str]:
    """
    Attempts to load patient/eye mapping from an Excel or CSV file.
    Supports OIMHS 'Demographics of the participants.xlsx' (Eye ID -> Patient ID).
    """
    mapping = {}
    csv_candidates = [metadata_path] if metadata_path else []
    if not metadata_path:
        csv_candidates += glob.glob(os.path.join(raw_dir, "**", "*.xlsx"), recursive=True)
        csv_candidates += glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)

    for path in csv_candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
            cols_lower = {str(c).lower().strip(): c for c in df.columns}

            # Check if this is OIMHS Demographics with 'Patient ID' and 'Eye ID'
            patient_col = next((cols_lower[c] for c in cols_lower if "patient" in c), None)
            eye_col = next((cols_lower[c] for c in cols_lower if "eye" in c and "category" not in c), None)

            if patient_col and eye_col and patient_col != eye_col:
                for _, row in df.iterrows():
                    e_id = str(row[eye_col]).strip()
                    p_id = str(row[patient_col]).strip()
                    # Map Eye ID to Patient ID to prevent cross-eye patient leakage
                    mapping[e_id] = f"patient_{p_id}"
                print(f"[INFO] Loaded {len(mapping)} Eye->Patient mappings from: {os.path.basename(path)}")
                break

            # Fallback for generic metadata tables
            img_col = next((cols_lower[c] for c in cols_lower if any(k in c for k in ["image", "img", "filename", "file"])), None)
            subj_col = next((cols_lower[c] for c in cols_lower if any(k in c for k in ["eye", "patient", "subject"])), None)
            if img_col and subj_col and img_col != subj_col:
                for _, row in df.iterrows():
                    img_name = os.path.splitext(os.path.basename(str(row[img_col])))[0]
                    mapping[img_name] = str(row[subj_col])
                print(f"[INFO] Loaded {len(mapping)} subject mappings from: {os.path.basename(path)}")
                break
        except Exception:
            continue
    return mapping


def extract_eye_id(filename: str, metadata_map: Dict[str, str], pattern: Optional[str] = None) -> str:
    """
    Extracts subject/patient identifier:
    1. Parent directory (e.g. Eye ID in OIMHS 'Images/<Eye_ID>/<Scan>.png') checked against metadata map.
    2. Filename base checked against metadata map.
    3. Custom regex pattern if provided.
    4. Base name or prefix before underscore.
    """
    # 1. Check parent directory (for OIMHS structure)
    parent_dir = os.path.basename(os.path.dirname(filename))
    if parent_dir in metadata_map:
        return metadata_map[parent_dir]

    # 2. Check filename base
    base = os.path.splitext(os.path.basename(filename))[0]
    if base in metadata_map:
        return metadata_map[base]

    # 3. Check regex pattern
    if pattern:
        match = re.search(pattern, base)
        if match:
            return match.group(1)

    # 4. Check underscore prefix
    parts = base.split("_")
    if len(parts) > 1:
        return parts[0]

    # Fallback to parent directory if non-trivial
    if parent_dir and parent_dir not in ["images", "Images", "raw"]:
        return parent_dir

    return base


def find_image_mask_pairs(
    raw_dir: str,
    img_subdir_candidates: List[str] = None,
    mask_subdir_candidates: List[str] = None,
) -> List[Tuple[str, str]]:
    if img_subdir_candidates is None:
        img_subdir_candidates = [
            "images", "Images", "Image", "Train/images", "Train/Images",
            "train/images", "train/Images", "GOALS2022-Train/Train/images",
        ]
    if mask_subdir_candidates is None:
        mask_subdir_candidates = [
            "Layer_Masks", "layer_masks", "masks", "Masks", "Train/Layer_Masks",
            "train/layer_masks", "Train/masks", "train/masks", "GOALS2022-Train/Train/Layer_Masks",
        ]

    img_dir = None
    for cand in img_subdir_candidates:
        full_path = os.path.join(raw_dir, cand)
        if os.path.isdir(full_path):
            img_dir = full_path
            break

    mask_dir = None
    for cand in mask_subdir_candidates:
        full_path = os.path.join(raw_dir, cand)
        if os.path.isdir(full_path):
            mask_dir = full_path
            break

    if img_dir is None or mask_dir is None:
        all_pngs = sorted(glob.glob(os.path.join(raw_dir, "**", "*.png"), recursive=True))
        all_jpgs = sorted(glob.glob(os.path.join(raw_dir, "**", "*.jpg"), recursive=True))
        all_files = all_pngs + all_jpgs

        imgs = [f for f in all_files if "mask" not in f.lower()]
        masks = [f for f in all_files if "mask" in f.lower()]

        # If no separate mask files found, check for OIMHS composite side-by-side images
        if not masks and imgs:
            print(f"[INFO] No separate mask files found; treating {len(imgs)} scans as composite images (OIMHS format).")
            return [(img, img) for img in imgs]

        pairs = []
        mask_dict = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}
        for img in imgs:
            name = os.path.splitext(os.path.basename(img))[0]
            if name in mask_dict:
                pairs.append((img, mask_dict[name]))
            elif f"{name}_mask" in mask_dict:
                pairs.append((img, mask_dict[f"{name}_mask"]))
        return pairs

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    pairs = []
    for img_path in img_files:
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            continue
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, base_name)
        if not os.path.exists(mask_path):
            root_name = os.path.splitext(base_name)[0]
            mask_path_png = os.path.join(mask_dir, f"{root_name}.png")
            if os.path.exists(mask_path_png):
                mask_path = mask_path_png
            else:
                continue
        pairs.append((img_path, mask_path))

    return pairs


def write_split_file(filepath: str, pairs: List[Tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for img, mask in pairs:
            f.write(f"{img},{mask}\n")


def split_dataset(
    pairs: List[Tuple[str, str]],
    metadata_map: Dict[str, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    k_folds: Optional[int] = None,
    seed: int = 42,
    eye_pattern: Optional[str] = None,
    out_dir: str = "data/splits",
) -> None:
    eye_groups: Dict[str, List[Tuple[str, str]]] = {}
    for img_path, mask_path in pairs:
        eye_id = extract_eye_id(img_path, metadata_map, eye_pattern)
        eye_groups.setdefault(eye_id, []).append((img_path, mask_path))

    n_images = len(pairs)
    unique_eyes = sorted(list(eye_groups.keys()))
    n_eyes = len(unique_eyes)

    print("\n--- Eye-Level Split Verification ---")
    print(f"Total scans found:      {n_images}")
    print(f"Total unique eye IDs:   {n_eyes}")

    if n_images == n_eyes:
        print(
            "\n[WARNING] Number of unique eye IDs EQUALS total images (each image has a unique ID).\n"
            "  The files/directory do not encode patient or eye groups in their names.\n"
            "  Unless each image in the dataset comes from a distinct subject/eye by design,\n"
            "  this split operates as an image-level split.\n"
            "  If patient metadata is available, supply it via --metadata_csv, or document this in the report.\n"
        )
    else:
        print(f"[INFO] Verified: {n_images} scans clustered into {n_eyes} unique eye subjects.")

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_eyes)

    # 1. Standard train / val / test split
    n_train = int(round(train_ratio * n_eyes))
    n_val = int(round(val_ratio * n_eyes))

    train_eyes = set(unique_eyes[:n_train])
    val_eyes = set(unique_eyes[n_train : n_train + n_val])
    test_eyes = set(unique_eyes[n_train + n_val :])

    train_pairs = [p for eye in train_eyes for p in eye_groups[eye]]
    val_pairs = [p for eye in val_eyes for p in eye_groups[eye]]
    test_pairs = [p for eye in test_eyes for p in eye_groups[eye]]

    write_split_file(os.path.join(out_dir, "train.txt"), train_pairs)
    write_split_file(os.path.join(out_dir, "val.txt"), val_pairs)
    write_split_file(os.path.join(out_dir, "test.txt"), test_pairs)

    print(f"Standard Split (seed={seed}):")
    print(f"  Train: {len(train_pairs)} scans ({len(train_eyes)} eyes) -> {out_dir}/train.txt")
    print(f"  Val:   {len(val_pairs)} scans ({len(val_eyes)} eyes) -> {out_dir}/val.txt")
    print(f"  Test:  {len(test_pairs)} scans ({len(test_eyes)} eyes) -> {out_dir}/test.txt")

    # 2. 5-Fold Cross-Validation by Eye (Nested 3-way split per fold)
    if k_folds and k_folds > 1:
        print(f"\nGenerating {k_folds}-Fold Cross-Validation splits by eye (3-way: train/val/test per fold)...")
        folds = np.array_split(unique_eyes, k_folds)
        for fold_idx in range(k_folds):
            # Held-out test fold: strictly for final reporting, never touched during training
            test_fold_eyes = set(folds[fold_idx])
            remaining_eyes = [e for e in unique_eyes if e not in test_fold_eyes]

            # 15% of remaining eyes form validation set for early stopping / best checkpoint selection
            n_rem = len(remaining_eyes)
            n_val = max(1, int(round(0.15 * n_rem))) if n_rem > 1 else 0
            val_fold_eyes = set(remaining_eyes[:n_val])
            train_fold_eyes = set(remaining_eyes[n_val:])

            f_train_pairs = [p for eye in train_fold_eyes for p in eye_groups[eye]]
            f_val_pairs = [p for eye in val_fold_eyes for p in eye_groups[eye]]
            f_test_pairs = [p for eye in test_fold_eyes for p in eye_groups[eye]]

            fold_dir = os.path.join(out_dir, f"cv_{k_folds}folds", f"fold_{fold_idx}")
            write_split_file(os.path.join(fold_dir, "train.txt"), f_train_pairs)
            write_split_file(os.path.join(fold_dir, "val.txt"), f_val_pairs)
            write_split_file(os.path.join(fold_dir, "test.txt"), f_test_pairs)
            print(
                f"  Fold {fold_idx}: Train {len(f_train_pairs)} scans ({len(train_fold_eyes)} eyes) | "
                f"Val {len(f_val_pairs)} scans ({len(val_fold_eyes)} eyes) | "
                f"Test {len(f_test_pairs)} scans ({len(test_fold_eyes)} eyes) -> {fold_dir}/"
            )


def main():
    parser = argparse.ArgumentParser(description="Split GOALS dataset by eye ID with leakage check.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw dataset folder.")
    parser.add_argument("--out_dir", type=str, default="data/splits", help="Directory to save split files.")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Optional metadata CSV/Excel with patient IDs.")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Fraction for training set.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction for validation set.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Fraction for testing set.")
    parser.add_argument("--cv_folds", "--k_folds", dest="cv_folds", type=int, default=5, help="Number of cross-validation folds by eye (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--eye_pattern", type=str, default=None, help="Optional regex to extract eye ID.")

    args = parser.parse_args()

    pairs = find_image_mask_pairs(args.raw_dir)
    if not pairs:
        print(f"No image-mask pairs found in '{args.raw_dir}'.")
        print("Please place GOALS images and Layer_Masks into 'data/raw/'.")
        return

    metadata_map = load_metadata_mapping(args.raw_dir, args.metadata_csv)
    split_dataset(
        pairs=pairs,
        metadata_map=metadata_map,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        k_folds=args.cv_folds,
        seed=args.seed,
        eye_pattern=args.eye_pattern,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
