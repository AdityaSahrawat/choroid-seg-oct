"""
Evaluation and visualization script for testing trained Choroid models.

Outputs:
- Primary clinical Bland-Altman agreement plot (ONE POINT PER SCAN)
- Secondary pooled-column Bland-Altman diagnostic plot
- Subfoveal Choroidal Thickness (SFCT) and Parafoveal thickness measurements
- Robustness Quality breakdown (Good vs. Poor quality scans via Quality Assessment.xlsx)
- Boundary-specific HD95 (Bruch's membrane HD95 vs. CSI HD95 vs. 2D Mask HD95)
- Automatic extraction of top 3 failure cases (ranked by CSI lower boundary error)
- Automatic extraction of top 3 worst HD95 cases (with analysis of edge drivers)
- Summary metrics saved to JSON and CSV
"""

import argparse
import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import ChoroidDataset
from src.evaluate import (
    compute_bland_altman,
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
    extract_pred_boundaries,
)
from src.preprocess import OCTPreprocessor
from src.thickness import (
    compute_foveal_thickness,
    compute_pixel_count_thickness,
    compute_thickness_profile,
    extract_macular_hole_center,
    find_foveal_scans_in_dataset,
    summarize_thickness,
)


def cv2_resize_mask(mask: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
    orig_h, orig_w = orig_hw
    return cv2.resize((mask > 0.5).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def plot_qualitative_result(
    img_gray: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    top_gt: np.ndarray,
    bot_gt: np.ndarray,
    top_pred: np.ndarray,
    bot_pred: np.ndarray,
    save_path: str,
    title: str = "Choroid Segmentation",
    fovea_col: Optional[int] = None,
) -> None:
    # Ensure grayscale image dimensions match mask dimensions (e.g. for OIMHS composite side-by-side scans)
    if img_gray.shape[:2] != gt_mask.shape[:2]:
        if img_gray.shape[1] >= gt_mask.shape[1]:
            img_gray = img_gray[:, :gt_mask.shape[1]]
        else:
            img_gray = cv2.resize(img_gray, (gt_mask.shape[1], gt_mask.shape[0]))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # 1. Raw B-scan
    axes[0].imshow(img_gray, cmap="gray")
    if fovea_col is not None:
        axes[0].axvline(fovea_col, color="cyan", linestyle="--", linewidth=1.5, label=f"Subfoveal Col: {fovea_col}")
        axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{title} - Raw B-Scan")
    axes[0].axis("off")

    # 2. Masks Comparison Overlay (Green: GT, Red: Pred, Yellow: Overlap)
    overlay = np.zeros((*img_gray.shape, 3), dtype=np.uint8)
    overlay[..., 0] = pred_mask * 255  # Red
    overlay[..., 1] = gt_mask * 255    # Green
    axes[1].imshow(img_gray, cmap="gray")
    axes[1].imshow(overlay, alpha=0.45)
    axes[1].set_title("Mask Overlay (Green: GT, Red: Pred, Yellow: Overlap)")
    axes[1].axis("off")

    # 3. Boundaries Overlay
    axes[2].imshow(img_gray, cmap="gray")
    cols = np.arange(len(top_pred))
    axes[2].plot(cols, top_gt, "g-", linewidth=1.5, label="GT Bruch's Membrane")
    axes[2].plot(cols, bot_gt, "g--", linewidth=1.5, label="GT CSI (Lower Boundary)")
    axes[2].plot(cols, top_pred, "r-", linewidth=1.5, label="Pred Bruch's Membrane")
    axes[2].plot(cols, bot_pred, "r--", linewidth=1.5, label="Pred CSI (Lower Boundary)")
    if fovea_col is not None:
        axes[2].axvline(fovea_col, color="cyan", linestyle=":", linewidth=1.5, label=f"Subfoveal (Col {fovea_col})")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("Extracted Boundaries (Bruch's Membrane & CSI)")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_thickness_profile(
    t_pred: np.ndarray,
    t_gt: np.ndarray,
    save_path: str,
    units: str = "px",
    sample_id: str = "Scan",
    fovea_col: Optional[int] = None,
) -> None:
    plt.figure(figsize=(9, 4.5))
    cols = np.arange(len(t_pred))
    plt.plot(cols, t_gt, label="Ground Truth Thickness", color="seagreen", linewidth=1.8)
    plt.plot(cols, t_pred, label="Predicted Thickness", color="crimson", linestyle="--", linewidth=1.8)
    if fovea_col is not None:
        plt.axvline(fovea_col, color="darkblue", linestyle=":", linewidth=1.5, label=f"Subfoveal Col: {fovea_col}")
    plt.xlabel("A-Scan Column Index")
    plt.ylabel(f"Choroidal Thickness [{units}]")
    plt.title(f"Column-wise Thickness Profile - {sample_id}")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bland_altman(
    pred_vals: np.ndarray,
    gt_vals: np.ndarray,
    save_path: str,
    units: str = "px",
    title: str = "Bland-Altman Plot (Per-Scan Means)",
) -> Dict[str, float]:
    diff = pred_vals - gt_vals
    mean = (pred_vals + gt_vals) / 2.0
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.7, s=25, color="royalblue", edgecolors="black", linewidth=0.5)
    plt.axhline(bias, color="crimson", linestyle="-", label=f"Mean Bias: {bias:.2f} {units}")
    plt.axhline(loa_high, color="black", linestyle="--", label=f"+1.96 SD: {loa_high:.2f} {units}")
    plt.axhline(loa_low, color="black", linestyle="--", label=f"-1.96 SD: {loa_low:.2f} {units}")
    plt.xlabel(f"Mean Thickness of GT and Pred [{units}]")
    plt.ylabel(f"Difference (Pred - GT) [{units}]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {"bias": bias, "sd": sd, "loa_low": loa_low, "loa_high": loa_high}


def load_quality_assessment(raw_dir: str = "data/raw/OIMHS dataset") -> Dict[Tuple[str, str], Dict[str, Any]]:
    qa_path = os.path.join(raw_dir, "Quality Assessment.xlsx")
    qa_map = {}
    if os.path.isfile(qa_path):
        try:
            df = pd.read_excel(qa_path)
            for _, row in df.iterrows():
                e_id = str(row["Eye ID"]).strip()
                img_id = str(row["Image ID"]).strip()
                low_sig = bool(row["Low signal strength"] == 1)
                shield = bool(row["Signal shield"] == 1)
                blur = bool(row["Image blur"] == 1)
                is_poor = bool(low_sig or shield or blur)
                qa_map[(e_id, img_id)] = {
                    "quality": "Poor" if is_poor else "Good",
                    "low_signal": low_sig,
                    "shield": shield,
                    "blur": blur,
                }
        except Exception as e:
            print(f"[WARNING] Failed to load Quality Assessment.xlsx: {e}")
    return qa_map


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained choroid segmentation model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--split_file", type=str, default="data/splits/test.txt", help="Path to split file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save figures and metrics")
    parser.add_argument("--num_visualizations", type=int, default=5, help="Number of qualitative plots to generate")
    parser.add_argument("--no_postprocess", action="store_true", help="Disable post-processing (ablation: raw mask, no LCC, no hole filling, no SavGol)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    exp_name = cfg.get("name", "experiment")

    default_subfolder = "test_eval_raw" if args.no_postprocess else "test_eval"
    out_dir = args.output_dir or os.path.join(cfg.get("output_dir", "outputs"), exp_name, default_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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

    p_cfg = cfg.get("preprocess", {})
    preproc = OCTPreprocessor(
        use_clahe=p_cfg.get("use_clahe", True),
        clip_limit=p_cfg.get("clip_limit", 2.0),
        tile_grid_size=tuple(p_cfg.get("tile_grid_size", [8, 8])),
        denoise_method=p_cfg.get("denoise_method", None),
    )

    d_cfg = cfg.get("data", {})
    img_paths, mask_paths = [], []
    with open(args.split_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                p = line.split(",")
                img_paths.append(p[0].strip())
                mask_paths.append(p[1].strip() if len(p) > 1 else None)

    dataset = ChoroidDataset(
        img_paths=img_paths,
        mask_paths=mask_paths,
        target_size=d_cfg.get("target_size", 512),
        choroid_pixel_val=d_cfg.get("choroid_pixel_val", 160),
        mask_tolerance=d_cfg.get("mask_tolerance", 0),
        is_train=False,
        preprocessor=preproc,
        normalize_mean=d_cfg.get("normalize_mean", 0.485),
        normalize_std=d_cfg.get("normalize_std", 0.229),
    )

    t_cfg = cfg.get("thickness", {})
    um_per_px = t_cfg.get("um_per_px", None)
    smooth_w = t_cfg.get("smooth_window", 31)
    poly_o = t_cfg.get("poly_order", 2)
    threshold = cfg.get("threshold", 0.5)

    # 1. Quality Assessment metadata mapping
    qa_map = load_quality_assessment()
    if qa_map:
        print(f"Loaded quality flags for {len(qa_map)} scans.")

    # 2. Fovea detection across test cohort
    fovea_map = find_foveal_scans_in_dataset(img_paths)
    print(f"Identified foveal scans across {len(fovea_map)} eyes in the evaluation set.")

    results_records = []
    per_scan_pred_means = []
    per_scan_gt_means = []
    pooled_pred_cols = []
    pooled_gt_cols = []

    foveal_scan_records = []
    cached_eval_items = []

    print(f"Evaluating {len(dataset)} scans on device '{device}'...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            x = item["image"].unsqueeze(0).to(device)
            orig_hw = item["orig_hw"]
            img_path = item["img_path"]
            sample_name = os.path.splitext(os.path.basename(img_path))[0]
            eye_id = os.path.basename(os.path.dirname(img_path))

            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            # Clean prediction mask (LCC-only when postprocessing is enabled, hole filling is OFF)
            if args.no_postprocess:
                pred_mask = clean_mask(prob, orig_hw=orig_hw, threshold=threshold, keep_largest_component=False, fill_holes=False)
            else:
                pred_mask = clean_mask(prob, orig_hw=orig_hw, threshold=threshold, keep_largest_component=True, fill_holes=False)

            has_gt = "mask" in item
            if has_gt:
                if "orig_mask" in item:
                    gt_mask = (item["orig_mask"] > 0).astype(np.uint8)
                else:
                    y = item["mask"][0].numpy()
                    gt_mask = cv2_resize_mask(y, orig_hw)

                dice_score, iou_score = compute_dice_iou(pred_mask, gt_mask)
                hd95_2d_mask = compute_hd95(pred_mask, gt_mask)

                # Ground truth strictly raw without smoothing or gap interpolation
                top_gt, bot_gt, valid_gt = extract_gt_boundaries(gt_mask)

                # Prediction boundaries (raw if no_postprocess or smooth_w is null/0)
                if args.no_postprocess or smooth_w is None or smooth_w == 0:
                    top_pred, bot_pred, valid_pred = extract_gt_boundaries(pred_mask)
                else:
                    top_pred, bot_pred, valid_pred = extract_pred_boundaries(
                        pred_mask, smooth_window=smooth_w, poly_order=poly_o
                    )

                b_errs = compute_boundary_errors(
                    top_pred, bot_pred, top_gt, bot_gt, um_per_px=um_per_px
                )

                t_pred, units = compute_thickness_profile(top_pred, bot_pred, valid_pred, um_per_px)
                t_gt, _ = compute_thickness_profile(top_gt, bot_gt, valid_gt, um_per_px)

                # Pixel-count thickness cross-check (direct column pixel summation)
                t_pred_pixel, _ = compute_pixel_count_thickness(pred_mask, um_per_px)
                t_gt_pixel, _ = compute_pixel_count_thickness(gt_mask, um_per_px)

                t_metrics = compute_thickness_metrics(t_pred, t_gt, units=units)
                t_metrics_pixel = compute_thickness_metrics(t_pred_pixel, t_gt_pixel, units=units)

                # Store per-scan means for independent clinical Bland-Altman
                if not np.isnan(t_metrics["mean_pred"]) and not np.isnan(t_metrics["mean_gt"]):
                    per_scan_pred_means.append(t_metrics["mean_pred"])
                    per_scan_gt_means.append(t_metrics["mean_gt"])

                # Store pooled columns for secondary diagnostic plot
                valid_pts = (~np.isnan(t_pred)) & (~np.isnan(t_gt))
                pooled_pred_cols.extend(t_pred[valid_pts].tolist())
                pooled_gt_cols.extend(t_gt[valid_pts].tolist())

                # Quality assessment lookup
                qa_info = qa_map.get((eye_id, sample_name), {"quality": "Unknown", "blur": False, "shield": False, "low_signal": False})

                # Check if this is the primary foveal B-scan for this eye
                is_foveal_scan = False
                c_fovea_val = None
                fov_info = fovea_map.get(eye_id, None)
                if fov_info is not None and fov_info["foveal_scan"] == img_path:
                    is_foveal_scan = True
                    c_fovea_val = fov_info["c_fovea"]

                # If this scan itself has a macular hole, get its hole column
                c_hole_scan, hole_area_scan = extract_macular_hole_center(img_path)

                record = {
                    "eye_id": eye_id,
                    "sample": sample_name,
                    "img_path": img_path,
                    "quality": qa_info["quality"],
                    "is_blur": qa_info["blur"],
                    "is_shield": qa_info["shield"],
                    "is_low_signal": qa_info["low_signal"],
                    "is_foveal_scan": is_foveal_scan,
                    "c_fovea": c_fovea_val,
                    "dice": dice_score,
                    "iou": iou_score,
                    "hd95_mask_px": hd95_2d_mask,
                    "bruchs_membrane_mae_px": b_errs["bruchs_membrane_mae_px"],
                    "csi_mae_px": b_errs["csi_mae_px"],
                    "hd95_bruchs_membrane_px": b_errs["hd95_bruchs_membrane_px"],
                    "hd95_csi_px": b_errs["hd95_csi_px"],
                    "ae95_bruchs_membrane_px": b_errs.get("ae95_bruchs_membrane_px", np.nan),
                    "ae95_csi_px": b_errs.get("ae95_csi_px", np.nan),
                    "thickness_mae": t_metrics["thickness_mae"],
                    "thickness_mae_pixel_count": t_metrics_pixel["thickness_mae"],
                    "thickness_rmse": t_metrics["thickness_rmse"],
                    "pearson_r": t_metrics["pearson_r"],
                    "mean_pred_thickness": t_metrics["mean_pred"],
                    "mean_gt_thickness": t_metrics["mean_gt"],
                    "units": units,
                }

                # Foveal measurements
                if is_foveal_scan and c_fovea_val is not None:
                    fov_thick = compute_foveal_thickness(t_pred, t_gt, c_fovea_val, window_radius=25, units=units)
                    fov_thick_px = compute_foveal_thickness(t_pred_pixel, t_gt_pixel, c_fovea_val, window_radius=25, units=units)
                    record["subfoveal_thickness_pred"] = fov_thick["subfoveal_thickness_pred"]
                    record["subfoveal_thickness_gt"] = fov_thick["subfoveal_thickness_gt"]
                    record["subfoveal_thickness_mae"] = fov_thick["subfoveal_thickness_mae"]
                    record["subfoveal_thickness_mae_pixel_count"] = fov_thick_px["subfoveal_thickness_mae"]
                    record["parafoveal_thickness_pred"] = fov_thick["parafoveal_thickness_pred"]
                    record["parafoveal_thickness_gt"] = fov_thick["parafoveal_thickness_gt"]
                    record["parafoveal_thickness_mae"] = fov_thick["parafoveal_thickness_mae"]
                    record["parafoveal_thickness_mae_pixel_count"] = fov_thick_px["parafoveal_thickness_mae"]
                    foveal_scan_records.append(record)

                results_records.append(record)

                cached_eval_items.append({
                    "sample_name": f"eye_{eye_id}_scan_{sample_name}",
                    "img_path": img_path,
                    "gt_mask": gt_mask,
                    "pred_mask": pred_mask,
                    "top_gt": top_gt,
                    "bot_gt": bot_gt,
                    "top_pred": top_pred,
                    "bot_pred": bot_pred,
                    "t_pred": t_pred,
                    "t_gt": t_gt,
                    "csi_mae_px": b_errs["csi_mae_px"],
                    "hd95_mask_px": hd95_2d_mask,
                    "fovea_col": c_fovea_val,
                })

                # Qualitative plots for first N samples
                if idx < args.num_visualizations:
                    img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    fig_path = os.path.join(out_dir, "figures", f"{eye_id}_{sample_name}_seg.png")
                    plot_qualitative_result(
                        img_raw, gt_mask, pred_mask,
                        top_gt, bot_gt, top_pred, bot_pred,
                        fig_path, title=f"Eye {eye_id} Scan {sample_name}",
                        fovea_col=c_fovea_val,
                    )
                    prof_path = os.path.join(out_dir, "figures", f"{eye_id}_{sample_name}_thickness.png")
                    plot_thickness_profile(
                        t_pred, t_gt, prof_path, units=units,
                        sample_id=f"Eye {eye_id} Scan {sample_name}",
                        fovea_col=c_fovea_val,
                    )

    df = pd.DataFrame(results_records)
    csv_path = os.path.join(out_dir, "test_metrics_per_scan.csv")
    df.to_csv(csv_path, index=False)

    # 1. Primary Bland-Altman agreement on PER-SCAN MEANS (independent samples)
    ba_per_scan = {}
    if len(per_scan_pred_means) > 1:
        ba_path = os.path.join(out_dir, "figures", "bland_altman_per_scan_means.png")
        ba_per_scan = plot_bland_altman(
            np.array(per_scan_pred_means),
            np.array(per_scan_gt_means),
            ba_path,
            units=units,
            title="Bland-Altman Agreement (One Point Per Scan)",
        )

    # 2. Secondary Bland-Altman on pooled columns (diagnostic only)
    ba_pooled = {}
    if len(pooled_pred_cols) > 0:
        ba_pooled_path = os.path.join(out_dir, "figures", "bland_altman_pooled_columns.png")
        ba_pooled = plot_bland_altman(
            np.array(pooled_pred_cols),
            np.array(pooled_gt_cols),
            ba_pooled_path,
            units=units,
            title="Diagnostic Bland-Altman Plot (Pooled Columns)",
        )

    # Cohort Pearson correlation across independent scan means
    if len(per_scan_pred_means) > 1:
        corr_mat = np.corrcoef(per_scan_pred_means, per_scan_gt_means)
        cohort_pearson_r = float(corr_mat[0, 1])
    else:
        cohort_pearson_r = float("nan")

    # Quality breakdown summary
    quality_summary = {}
    for q_label in ["Good", "Poor"]:
        sub_df = df[df["quality"] == q_label]
        if len(sub_df) > 0:
            quality_summary[q_label] = {
                "n_scans": len(sub_df),
                "mean_dice": float(sub_df["dice"].mean()),
                "std_dice": float(sub_df["dice"].std()),
                "mean_iou": float(sub_df["iou"].mean()),
                "mean_bruchs_membrane_mae_px": float(sub_df["bruchs_membrane_mae_px"].mean()),
                "mean_csi_mae_px": float(sub_df["csi_mae_px"].mean()),
                "mean_thickness_mae": float(sub_df["thickness_mae"].mean()),
                "mean_hd95_mask_px": float(sub_df["hd95_mask_px"].mean()),
            }

    # Subfoveal / Parafoveal summary across foveal scans
    foveal_summary = {}
    if foveal_scan_records:
        f_df = pd.DataFrame(foveal_scan_records)
        foveal_summary = {
            "n_foveal_eyes": len(f_df),
            "mean_subfoveal_gt": float(f_df["subfoveal_thickness_gt"].mean()),
            "mean_subfoveal_pred": float(f_df["subfoveal_thickness_pred"].mean()),
            "mean_subfoveal_mae": float(f_df["subfoveal_thickness_mae"].mean()),
            "mean_subfoveal_mae_pixel_count": float(f_df["subfoveal_thickness_mae_pixel_count"].mean()),
            "mean_parafoveal_gt": float(f_df["parafoveal_thickness_gt"].mean()),
            "mean_parafoveal_pred": float(f_df["parafoveal_thickness_pred"].mean()),
            "mean_parafoveal_mae": float(f_df["parafoveal_thickness_mae"].mean()),
            "mean_parafoveal_mae_pixel_count": float(f_df["parafoveal_thickness_mae_pixel_count"].mean()),
        }

    summary = {
        "postprocessing_applied": not args.no_postprocess,
        "n_scans": len(df),
        "mean_dice": float(df["dice"].mean()),
        "std_dice": float(df["dice"].std()),
        "mean_iou": float(df["iou"].mean()),
        "std_iou": float(df["iou"].std()),
        "mean_bruchs_membrane_mae_px": float(df["bruchs_membrane_mae_px"].mean()),
        "mean_csi_mae_px": float(df["csi_mae_px"].mean()),
        "mean_thickness_mae": float(df["thickness_mae"].mean()),
        "mean_hd95_mask_px": float(df["hd95_mask_px"].mean()),
        "mean_hd95_bruchs_membrane_px": float(df["hd95_bruchs_membrane_px"].mean()),
        "mean_hd95_csi_px": float(df["hd95_csi_px"].mean()),
        "mean_ae95_bruchs_membrane_px": float(df["ae95_bruchs_membrane_px"].mean()),
        "mean_ae95_csi_px": float(df["ae95_csi_px"].mean()),
        "cohort_pearson_r_per_scan": cohort_pearson_r,
        "bland_altman_per_scan": ba_per_scan,
        "bland_altman_pooled_columns": ba_pooled,
        "quality_breakdown": quality_summary,
        "foveal_summary": foveal_summary,
        "notes": {
            "fovea_located_from_gt_hole_mask": True,
            "fovea_location_statement": "Fovea located from reference macular-hole annotations (red mask BGR [0,0,255]); clinical application to unannotated scans requires joint multi-task hole segmentation.",
            "axial_scale_assumption": "um_per_px = 3.87 * 496 / 512 (~3.75 um/px) is an unconfirmed Spectralis nominal assumption; not encoded in released PNGs.",
            "lateral_scale_limitation": "A-scan column window (51 cols) covers variable physical width across eyes due to pre-release resampling of 384/512/768 raw widths to 512 cols.",
        },
        "units": units,
    }

    with open(os.path.join(out_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Failure-Case Visualizations (top 3 worst scans by lower boundary / CSI error)
    if cached_eval_items:
        cached_eval_items.sort(key=lambda item: item["csi_mae_px"], reverse=True)
        failure_cases = cached_eval_items[:3]
        fail_dir = os.path.join(out_dir, "failure_cases")
        os.makedirs(fail_dir, exist_ok=True)
        for rank, f_item in enumerate(failure_cases, start=1):
            img_raw = cv2.imread(f_item["img_path"], cv2.IMREAD_GRAYSCALE)
            s_name = f_item["sample_name"]
            fail_fig_path = os.path.join(fail_dir, f"rank{rank}_worst_CSI_{s_name}.png")
            plot_qualitative_result(
                img_raw, f_item["gt_mask"], f_item["pred_mask"],
                f_item["top_gt"], f_item["bot_gt"], f_item["top_pred"], f_item["bot_pred"],
                fail_fig_path,
                title=f"Worst Case #{rank} ({s_name}, CSI MAE: {f_item['csi_mae_px']:.2f} {units})",
                fovea_col=f_item.get("fovea_col"),
            )
        print(f"Saved {len(failure_cases)} worst-case CSI failure visualizations to {fail_dir}/")

    # Worst HD95 Visualizations (top 3 worst scans by mask HD95)
    if cached_eval_items:
        cached_eval_items.sort(key=lambda item: item["hd95_mask_px"] if not np.isnan(item["hd95_mask_px"]) else -1, reverse=True)
        hd95_worst = cached_eval_items[:3]
        hd95_dir = os.path.join(out_dir, "hd95_worst_cases")
        os.makedirs(hd95_dir, exist_ok=True)
        for rank, f_item in enumerate(hd95_worst, start=1):
            img_raw = cv2.imread(f_item["img_path"], cv2.IMREAD_GRAYSCALE)
            s_name = f_item["sample_name"]
            hd95_fig_path = os.path.join(hd95_dir, f"rank{rank}_worst_HD95_{s_name}.png")
            plot_qualitative_result(
                img_raw, f_item["gt_mask"], f_item["pred_mask"],
                f_item["top_gt"], f_item["bot_gt"], f_item["top_pred"], f_item["bot_pred"],
                hd95_fig_path,
                title=f"Worst HD95 #{rank} ({s_name}, Mask HD95: {f_item['hd95_mask_px']:.2f} px)",
                fovea_col=f_item.get("fovea_col"),
            )
        print(f"Saved {len(hd95_worst)} worst-case HD95 visualizations to {hd95_dir}/")

    print("\n" + "=" * 80)
    print("TEST EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Scans Evaluated:              {len(df)}")
    print(f"Post-Processing Enabled:      {not args.no_postprocess}")
    print(f"Mean Dice Similarity (DSC):   {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
    print(f"Mean IoU:                     {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
    print(f"Bruch's Membrane MAE:         {summary['mean_bruchs_membrane_mae_px']:.3f} px")
    print(f"CSI (Lower Boundary) MAE:     {summary['mean_csi_mae_px']:.3f} px")
    print(f"Choroidal Thickness MAE:      {summary['mean_thickness_mae']:.3f} px")
    print(f"Mask Contour HD95 (2D):       {summary['mean_hd95_mask_px']:.2f} px")
    print(f"Bruch's Membrane HD95 (1D):   {summary['mean_hd95_bruchs_membrane_px']:.2f} px")
    print(f"CSI Lower Boundary HD95 (1D): {summary['mean_hd95_csi_px']:.2f} px")
    print(f"Per-Scan Pearson r:           {summary['cohort_pearson_r_per_scan']:.4f}")
    print(f"Bland-Altman Bias (Per-Scan): {ba_per_scan.get('bias', float('nan')):.3f} px (95% LoA: [{ba_per_scan.get('loa_low', float('nan')):.2f}, {ba_per_scan.get('loa_high', float('nan')):.2f}] px)")

    if quality_summary:
        print("\n--- Quality Robustness Breakdown ---")
        for q_label, q_stats in quality_summary.items():
            anecdotal = " (Anecdotal: n < 10)" if q_stats["n_scans"] < 10 else ""
            print(f"  {q_label} Quality (n={q_stats['n_scans']}{anecdotal}): Dice={q_stats['mean_dice']:.4f}, BM MAE={q_stats['mean_bruchs_membrane_mae_px']:.2f} px, CSI MAE={q_stats['mean_csi_mae_px']:.2f} px, Thickness MAE={q_stats['mean_thickness_mae']:.2f} px")

    if foveal_summary:
        print("\n--- Subfoveal Choroidal Thickness (SFCT) Across Foveal Scans ---")
        print(f"  Foveal Eyes Evaluated:        {foveal_summary['n_foveal_eyes']}")
        print(f"  Mean Subfoveal GT Thickness:  {foveal_summary['mean_subfoveal_gt']:.2f} px")
        print(f"  Mean Subfoveal Pred Thickness:{foveal_summary['mean_subfoveal_pred']:.2f} px")
        print(f"  Subfoveal Thickness MAE (Main Run):   {foveal_summary['mean_subfoveal_mae']:.2f} px")
        print(f"  Subfoveal MAE (Pixel Count Check):    {foveal_summary['mean_subfoveal_mae_pixel_count']:.2f} px")
        print(f"  Parafoveal Thickness MAE (Main Run):  {foveal_summary['mean_parafoveal_mae']:.2f} px")
        print(f"  Parafoveal MAE (Pixel Count Check):   {foveal_summary['mean_parafoveal_mae_pixel_count']:.2f} px")

    print(f"\nDetailed metrics saved to:\n  - {csv_path}\n  - {out_dir}/test_summary.json")


if __name__ == "__main__":
    main()
