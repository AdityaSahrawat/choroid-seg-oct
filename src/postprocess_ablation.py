"""
Step-by-step post-processing ablation study for choroid OCT segmentation.

Evaluates 7 distinct configurations on the exact same model checkpoint:
1. raw: Direct probability thresholding at native resolution (no LCC, no hole filling, no smoothing)
2. lcc_only: Largest connected component only
3. lcc_hole: LCC + binary hole filling
4. savgol_w7: LCC + hole fill + Savitzky-Golay (window=7, poly=2)
5. savgol_w11: LCC + hole fill + Savitzky-Golay (window=11, poly=2)
6. savgol_w15: LCC + hole fill + Savitzky-Golay (window=15, poly=2)
7. savgol_w31: LCC + hole fill + Savitzky-Golay (window=31, poly=2)

Performs paired Wilcoxon signed-rank tests per scan against the Raw baseline.
"""

import argparse
import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

from src.dataset import ChoroidDataset
from src.evaluate import (
    compute_boundary_errors,
    compute_curve_hd95,
    compute_dice_iou,
    compute_hd95,
    compute_thickness_metrics,
)
from src.models import build_model
from src.postprocess import (
    clean_mask,
    extract_boundaries,
    extract_gt_boundaries,
)
from src.preprocess import OCTPreprocessor
from src.thickness import compute_thickness_profile


CONFIGURATIONS = [
    {
        "id": "raw",
        "name": "Raw (No post-processing)",
        "keep_lcc": False,
        "fill_holes": False,
        "smooth_w": None,
        "interp": False,
    },
    {
        "id": "lcc_only",
        "name": "LCC only",
        "keep_lcc": True,
        "fill_holes": False,
        "smooth_w": None,
        "interp": False,
    },
    {
        "id": "lcc_hole",
        "name": "LCC + Hole Fill",
        "keep_lcc": True,
        "fill_holes": True,
        "smooth_w": None,
        "interp": False,
    },
    {
        "id": "savgol_w7",
        "name": "LCC + Hole Fill + SavGol (w=7)",
        "keep_lcc": True,
        "fill_holes": True,
        "smooth_w": 7,
        "interp": True,
    },
    {
        "id": "savgol_w11",
        "name": "LCC + Hole Fill + SavGol (w=11)",
        "keep_lcc": True,
        "fill_holes": True,
        "smooth_w": 11,
        "interp": True,
    },
    {
        "id": "savgol_w15",
        "name": "LCC + Hole Fill + SavGol (w=15)",
        "keep_lcc": True,
        "fill_holes": True,
        "smooth_w": 15,
        "interp": True,
    },
    {
        "id": "savgol_w31",
        "name": "LCC + Hole Fill + SavGol (w=31)",
        "keep_lcc": True,
        "fill_holes": True,
        "smooth_w": 31,
        "interp": True,
    },
]


def run_ablation(checkpoint_path: str, split_file: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    m_cfg = cfg.get("model", {})
    model = build_model(
        model_type=m_cfg.get("model_type", "unet"),
        encoder_name=m_cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=m_cfg.get("in_channels", 1),
        classes=m_cfg.get("classes", 1),
        decoder_attention_type=m_cfg.get("decoder_attention_type", None),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    prep_cfg = cfg.get("preprocess", {})
    preproc = OCTPreprocessor(
        use_clahe=prep_cfg.get("use_clahe", True),
        clip_limit=prep_cfg.get("clip_limit", 2.0),
        tile_grid_size=tuple(prep_cfg.get("tile_grid_size", [8, 8])),
        denoise_method=prep_cfg.get("denoise_method", None),
    )

    data_cfg = cfg.get("data", {})
    img_paths, mask_paths = [], []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                p = line.split(",")
                img_paths.append(p[0].strip())
                mask_paths.append(p[1].strip() if len(p) > 1 else None)

    dataset = ChoroidDataset(
        img_paths=img_paths,
        mask_paths=mask_paths,
        target_size=data_cfg.get("target_size", 512),
        choroid_pixel_val=data_cfg.get("choroid_pixel_val", 160),
        mask_tolerance=data_cfg.get("mask_tolerance", 0),
        is_train=False,
        preprocessor=preproc,
        normalize_mean=data_cfg.get("normalize_mean", 0.485),
        normalize_std=data_cfg.get("normalize_std", 0.229),
    )

    threshold = cfg.get("threshold", 0.5)
    t_cfg = cfg.get("thickness", {})
    um_per_px = t_cfg.get("um_per_px", None)

    # 1. First, cache raw model predictions for all images
    print(f"Inference on {len(dataset)} scans...")
    cached_probs = []
    cached_gt = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            x = item["image"].unsqueeze(0).to(device)
            orig_hw = item["orig_hw"]
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            cached_probs.append((prob, orig_hw, item["img_path"]))

            if "orig_mask" in item:
                gt_mask = (item["orig_mask"] > 0).astype(np.uint8)
            else:
                y = item["mask"][0].numpy()
                gt_mask = cv2.resize((y > 0.5).astype(np.uint8), (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)
            cached_gt.append(gt_mask)

    # 2. Evaluate each post-processing configuration
    print(f"Evaluating {len(CONFIGURATIONS)} post-processing configurations...")
    step_results = {}
    per_scan_metrics = {c["id"]: [] for c in CONFIGURATIONS}

    for step_cfg in CONFIGURATIONS:
        step_id = step_cfg["id"]
        step_name = step_cfg["name"]
        keep_lcc = step_cfg["keep_lcc"]
        fill_holes = step_cfg["fill_holes"]
        smooth_w = step_cfg["smooth_w"]
        interp = step_cfg["interp"]

        dice_list, iou_list = [], []
        bm_mae_list, csi_mae_list = [], []
        hd95_mask_list = []
        hd95_bm_list, hd95_csi_list = [], []
        thick_mae_list = []
        pooled_pred_cols, pooled_gt_cols = [], []

        for idx in range(len(cached_probs)):
            prob, orig_hw, img_p = cached_probs[idx]
            gt_mask = cached_gt[idx]

            # Generate binary mask according to configuration
            pred_mask = clean_mask(
                prob,
                orig_hw=orig_hw,
                threshold=threshold,
                keep_largest_component=keep_lcc,
                fill_holes=fill_holes,
            )

            # Mask overlap & 2D contour HD95
            d, iou = compute_dice_iou(pred_mask, gt_mask)
            hd95_2d = compute_hd95(pred_mask, gt_mask)

            # Ground truth raw boundaries (strictly un-smoothed and un-interpolated)
            top_gt, bot_gt, valid_gt = extract_gt_boundaries(gt_mask)

            # Prediction boundaries
            top_pred, bot_pred, valid_pred = extract_boundaries(
                pred_mask,
                smooth_window=smooth_w,
                poly_order=2,
                interpolate_gaps=interp,
            )

            b_errs = compute_boundary_errors(top_pred, bot_pred, top_gt, bot_gt, um_per_px=um_per_px)
            t_pred, units = compute_thickness_profile(top_pred, bot_pred, valid_pred, um_per_px)
            t_gt, _ = compute_thickness_profile(top_gt, bot_gt, valid_gt, um_per_px)
            t_metrics = compute_thickness_metrics(t_pred, t_gt, units=units)

            hd95_bm = compute_curve_hd95(top_pred, top_gt)
            hd95_csi = compute_curve_hd95(bot_pred, bot_gt)

            valid_cols = (~np.isnan(t_pred)) & (~np.isnan(t_gt))
            pooled_pred_cols.extend(t_pred[valid_cols].tolist())
            pooled_gt_cols.extend(t_gt[valid_cols].tolist())

            scan_rec = {
                "dice": d,
                "iou": iou,
                "hd95_mask": hd95_2d,
                "bm_mae": b_errs["bruchs_membrane_mae_px"],
                "csi_mae": b_errs["csi_mae_px"],
                "hd95_bm": hd95_bm,
                "hd95_csi": hd95_csi,
                "thickness_mae": t_metrics["thickness_mae"],
            }
            per_scan_metrics[step_id].append(scan_rec)

            dice_list.append(d)
            iou_list.append(iou)
            if not np.isnan(hd95_2d): hd95_mask_list.append(hd95_2d)
            bm_mae_list.append(b_errs["bruchs_membrane_mae_px"])
            csi_mae_list.append(b_errs["csi_mae_px"])
            if not np.isnan(hd95_bm): hd95_bm_list.append(hd95_bm)
            if not np.isnan(hd95_csi): hd95_csi_list.append(hd95_csi)
            thick_mae_list.append(t_metrics["thickness_mae"])

        # Pooled column difference SD
        diff_cols = np.array(pooled_pred_cols) - np.array(pooled_gt_cols)
        col_sd = float(np.std(diff_cols, ddof=1)) if len(diff_cols) > 1 else 0.0

        step_results[step_id] = {
            "name": step_name,
            "mean_dice": float(np.nanmean(dice_list)),
            "std_dice": float(np.nanstd(dice_list)),
            "mean_iou": float(np.nanmean(iou_list)),
            "mean_bm_mae_px": float(np.nanmean(bm_mae_list)),
            "mean_csi_mae_px": float(np.nanmean(csi_mae_list)),
            "mean_thickness_mae_px": float(np.nanmean(thick_mae_list)),
            "mean_hd95_mask_px": float(np.nanmean(hd95_mask_list)) if hd95_mask_list else float("nan"),
            "mean_hd95_bm_px": float(np.nanmean(hd95_bm_list)) if hd95_bm_list else float("nan"),
            "mean_hd95_csi_px": float(np.nanmean(hd95_csi_list)) if hd95_csi_list else float("nan"),
            "pooled_col_sd_px": col_sd,
        }

    # Load Demographics to map scan -> patient_id for independent patient-level testing
    demo_path = "data/raw/OIMHS dataset/Demographics of the participants.xlsx"
    eye_to_patient = {}
    if os.path.isfile(demo_path):
        try:
            demo_df = pd.read_excel(demo_path)
            eye_to_patient = {str(r["Eye ID"]).strip(): str(r["Patient ID"]).strip() for _, r in demo_df.iterrows()}
        except Exception as e:
            print(f"[WARNING] Failed to load Demographics: {e}")

    patient_ids = [eye_to_patient.get(p.split("/")[-2], p.split("/")[-2]) for p in img_paths]
    unique_patients = sorted(list(set(patient_ids)))
    n_pts = len(unique_patients)
    print(f"Cohort includes {len(img_paths)} scans across {n_pts} independent patients.")

    # 3. Paired Wilcoxon Signed-Rank Test on PER-PATIENT MEANS (Independent statistical units)
    raw_id = "raw"
    def get_patient_means(step_id, metric_key):
        vals = [r[metric_key] for r in per_scan_metrics[step_id]]
        pt_means = []
        for pt in unique_patients:
            pt_indices = [i for i, p_id in enumerate(patient_ids) if p_id == pt]
            pt_mean = float(np.mean([vals[i] for i in pt_indices]))
            pt_means.append(pt_mean)
        return np.array(pt_means)

    raw_thick_pt = get_patient_means(raw_id, "thickness_mae")
    raw_bm_pt = get_patient_means(raw_id, "bm_mae")
    raw_csi_pt = get_patient_means(raw_id, "csi_mae")
    raw_dice_pt = get_patient_means(raw_id, "dice")

    wilcoxon_tests = {}
    for step_cfg in CONFIGURATIONS:
        step_id = step_cfg["id"]
        if step_id == raw_id:
            continue

        s_thick_pt = get_patient_means(step_id, "thickness_mae")
        s_bm_pt = get_patient_means(step_id, "bm_mae")
        s_csi_pt = get_patient_means(step_id, "csi_mae")
        s_dice_pt = get_patient_means(step_id, "dice")

        def run_wilcox(a, b):
            diff = a - b
            if np.all(diff == 0):
                return {"stat": 0.0, "p_value": 1.0, "identical": True, "p_str": "identical to raw"}
            if len(a) < 5:
                # With n < 5 (e.g. pilot with only 2 patients), exact rank sums cannot compute asymptotic p-value
                return {"stat": 0.0, "p_value": 1.0, "identical": False, "p_str": "n<5 patients (pilot)"}
            try:
                res = wilcoxon(a, b, zero_method="wilcox")
                p_val = float(res.pvalue)
                if p_val < 0.001:
                    p_str = "p < 0.001*"
                elif p_val < 0.05:
                    p_str = f"p = {p_val:.4f}*"
                else:
                    p_str = f"p = {p_val:.4f} (ns)"
                return {"stat": float(res.statistic), "p_value": p_val, "identical": False, "p_str": p_str}
            except Exception as e:
                return {"stat": float("nan"), "p_value": float("nan"), "error": str(e), "p_str": "error"}

        wilcoxon_tests[step_id] = {
            "thickness_mae_vs_raw": run_wilcox(s_thick_pt, raw_thick_pt),
            "bm_mae_vs_raw": run_wilcox(s_bm_pt, raw_bm_pt),
            "csi_mae_vs_raw": run_wilcox(s_csi_pt, raw_csi_pt),
            "dice_vs_raw": run_wilcox(s_dice_pt, raw_dice_pt),
        }

    # Output table
    print("\n" + "=" * 105)
    print(f"STEP-BY-STEP POST-PROCESSING ABLATION RESULTS ({len(dataset)} Scans across {n_pts} Patients)")
    print("=" * 105)
    hdr = f"{'Configuration':<32} | {'Dice':<7} | {'BM MAE':<7} | {'CSI MAE':<8} | {'Thick MAE':<9} | {'Col SD':<7} | {'Wilcoxon (Patient)':<18}"
    print(hdr)
    print("-" * 105)

    for step_cfg in CONFIGURATIONS:
        s_id = step_cfg["id"]
        res = step_results[s_id]
        if s_id == raw_id:
            p_str = "Baseline (Raw)"
        else:
            p_str = wilcoxon_tests[s_id]["thickness_mae_vs_raw"].get("p_str", "")

        line = (
            f"{res['name']:<32} | "
            f"{res['mean_dice']:.4f}  | "
            f"{res['mean_bm_mae_px']:.3f} px | "
            f"{res['mean_csi_mae_px']:.3f} px  | "
            f"{res['mean_thickness_mae_px']:.3f} px   | "
            f"{res['pooled_col_sd_px']:.2f} px | "
            f"{p_str:<18}"
        )
        print(line)
    print("=" * 105)

    out_summary = {
        "step_results": step_results,
        "wilcoxon_tests": wilcoxon_tests,
    }
    with open(os.path.join(out_dir, "postprocess_ablation_summary.json"), "w") as f:
        json.dump(out_summary, f, indent=2)

    return out_summary


def main():
    parser = argparse.ArgumentParser(description="Run step-by-step postprocessing ablation study.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--split_file", type=str, default="data/splits/pilot/test.txt", help="Path to test split file")
    parser.add_argument("--output_dir", type=str, default="outputs/oimhs_pilot/ablation_postprocess", help="Output directory")
    args = parser.parse_args()

    run_ablation(args.checkpoint, args.split_file, args.output_dir)


if __name__ == "__main__":
    main()
