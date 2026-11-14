"""
Comprehensive Evaluation Metrics for Choroid Segmentation and Thickness Measurement.

Evaluates at three clinical levels:
1. Mask Overlap:
   - Dice Similarity Coefficient (DSC)
   - Intersection over Union (IoU / Jaccard)
   - 95th-percentile Hausdorff Distance (HD95)
2. Boundary Localization:
   - Upper choroid boundary Mean Absolute Error (MAE)
   - Lower choroid boundary (CSI) Mean Absolute Error (MAE)
3. Thickness Measurement Agreement:
   - Per-scan mean thickness comparison (primary clinical headline agreement)
   - Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
   - Pearson Correlation Coefficient (r) at both per-scan and cohort levels
   - Bland-Altman agreement: Mean Bias, Standard Deviation, and 95% Limits of Agreement
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.spatial.distance import cdist


def compute_dice_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """Computes Dice score and Intersection over Union (IoU)."""
    p = (pred_mask > 0).astype(np.bool_)
    g = (gt_mask > 0).astype(np.bool_)

    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    p_sum = p.sum()
    g_sum = g.sum()

    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0

    dice = (2.0 * intersection) / (p_sum + g_sum + eps)
    iou = intersection / (union + eps)
    return float(dice), float(iou)


def compute_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Computes 95th-percentile Hausdorff Distance (HD95) between mask contours in pixels.
    """
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    if p.sum() == 0 or g.sum() == 0:
        return float("nan")

    contours_p, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_g, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours_p or not contours_g:
        return float("nan")

    pts_p = np.vstack([c.squeeze(axis=1) for c in contours_p if len(c.squeeze(axis=1).shape) == 2])
    pts_g = np.vstack([c.squeeze(axis=1) for c in contours_g if len(c.squeeze(axis=1).shape) == 2])

    if pts_p.size == 0 or pts_g.size == 0:
        return float("nan")

    dists = cdist(pts_p, pts_g, metric="euclidean")
    min_dist_p_to_g = dists.min(axis=1)
    min_dist_g_to_p = dists.min(axis=0)

    hd95 = max(
        np.percentile(min_dist_p_to_g, 95),
        np.percentile(min_dist_g_to_p, 95),
    )
    return float(hd95)


def compute_curve_hd95(
    curve_pred: np.ndarray,
    curve_gt: np.ndarray,
) -> float:
    """
    Computes 95th-percentile Hausdorff Distance (HD95) specifically between two 1D boundary curves
    as 2D point sets (c, y(c)) in the B-scan plane.
    Points are parameterized as (c, y(c)) for valid columns.
    Metric is identical to 2D mask HD95: max(P95(min Euclidean d_A->B), P95(min Euclidean d_B->A)).
    """
    cols_p = np.where(~np.isnan(curve_pred))[0]
    cols_g = np.where(~np.isnan(curve_gt))[0]
    if len(cols_p) == 0 or len(cols_g) == 0:
        return float("nan")

    pts_p = np.column_stack([cols_p, curve_pred[cols_p]])
    pts_g = np.column_stack([cols_g, curve_gt[cols_g]])

    dists = cdist(pts_p, pts_g, metric="euclidean")
    min_p_to_g = dists.min(axis=1)
    min_g_to_p = dists.min(axis=0)

    hd95 = max(
        np.percentile(min_p_to_g, 95),
        np.percentile(min_g_to_p, 95),
    )
    return float(hd95)


def compute_curve_ae95(
    curve_pred: np.ndarray,
    curve_gt: np.ndarray,
) -> float:
    """
    Computes 95th-percentile Absolute Error (AE95) across co-registered valid columns.
    Measures vertical boundary localization error strictly without lateral horizontal column penalty.
    """
    valid = (~np.isnan(curve_pred)) & (~np.isnan(curve_gt))
    if not np.any(valid):
        return float("nan")
    errs = np.abs(curve_pred[valid] - curve_gt[valid])
    return float(np.percentile(errs, 95))


def compute_boundary_errors(
    top_pred: np.ndarray,
    bot_pred: np.ndarray,
    top_gt: np.ndarray,
    bot_gt: np.ndarray,
    um_per_px: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes Mean Absolute Error (MAE) and 95th-percentile errors for Bruch's membrane and CSI boundaries.
    Top: Bruch's membrane (Upper boundary).
    Bot: Choroid-Sclera Interface (CSI / Lower boundary).
    """
    valid_top = (~np.isnan(top_pred)) & (~np.isnan(top_gt))
    valid_bot = (~np.isnan(bot_pred)) & (~np.isnan(bot_gt))

    top_err = np.abs(top_pred[valid_top] - top_gt[valid_top]) if np.any(valid_top) else np.array([np.nan])
    bot_err = np.abs(bot_pred[valid_bot] - bot_gt[valid_bot]) if np.any(valid_bot) else np.array([np.nan])

    upper_mae_px = float(np.nanmean(top_err))
    lower_mae_px = float(np.nanmean(bot_err))

    # Curve-specific 2D Euclidean HD95
    hd95_bm_px = compute_curve_hd95(top_pred, top_gt)
    hd95_csi_px = compute_curve_hd95(bot_pred, bot_gt)

    # Column-wise vertical AE95 (no lateral penalty)
    ae95_bm_px = compute_curve_ae95(top_pred, top_gt)
    ae95_csi_px = compute_curve_ae95(bot_pred, bot_gt)

    res = {
        "upper_mae_px": upper_mae_px,
        "lower_mae_px": lower_mae_px,
        "bruchs_membrane_mae_px": upper_mae_px,
        "csi_mae_px": lower_mae_px,
        "hd95_bruchs_membrane_px": hd95_bm_px,
        "hd95_csi_px": hd95_csi_px,
        "ae95_bruchs_membrane_px": ae95_bm_px,
        "ae95_csi_px": ae95_csi_px,
    }

    if um_per_px is not None and um_per_px > 0:
        res["upper_mae_um"] = upper_mae_px * um_per_px
        res["lower_mae_um"] = lower_mae_px * um_per_px
        res["bm_mae_um"] = res["upper_mae_um"]
        res["csi_mae_um"] = res["lower_mae_um"]
        res["hd95_bm_um"] = hd95_bm_px * um_per_px
        res["hd95_csi_um"] = hd95_csi_px * um_per_px
        res["ae95_bm_um"] = ae95_bm_px * um_per_px
        res["ae95_csi_um"] = ae95_csi_px * um_per_px

    return res


def compute_thickness_metrics(
    t_pred: np.ndarray,
    t_gt: np.ndarray,
    units: str = "px",
) -> Dict[str, Any]:
    """
    Computes MAE, RMSE, Pearson correlation on valid columns of a single scan.
    """
    valid = (~np.isnan(t_pred)) & (~np.isnan(t_gt))
    if not np.any(valid):
        return {
            "thickness_mae": np.nan,
            "thickness_rmse": np.nan,
            "pearson_r": np.nan,
            "mean_pred": np.nan,
            "mean_gt": np.nan,
            "units": units,
        }

    p = t_pred[valid]
    g = t_gt[valid]

    diff = p - g
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if len(p) > 1 and np.std(p) > 1e-6 and np.std(g) > 1e-6:
        r = float(np.corrcoef(p, g)[0, 1])
    else:
        r = np.nan

    return {
        "thickness_mae": mae,
        "thickness_rmse": rmse,
        "pearson_r": r,
        "mean_pred": float(np.mean(p)),
        "mean_gt": float(np.mean(g)),
        "units": units,
    }


def compute_bland_altman(
    pred_vals: np.ndarray,
    gt_vals: np.ndarray,
) -> Dict[str, float]:
    """
    Computes standard Bland-Altman agreement statistics:
    Mean Bias, Standard Deviation, and 95% Limits of Agreement.
    Best computed on independent per-scan mean thickness values across the cohort.
    """
    valid = (~np.isnan(pred_vals)) & (~np.isnan(gt_vals))
    p = pred_vals[valid]
    g = gt_vals[valid]

    if len(p) == 0:
        return {
            "bias": np.nan,
            "sd": np.nan,
            "loa_low": np.nan,
            "loa_high": np.nan,
        }

    diff = p - g
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    return {
        "bias": bias,
        "sd": sd,
        "loa_low": loa_low,
        "loa_high": loa_high,
    }
