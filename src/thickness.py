"""
Choroidal Thickness Computation.

Clinical definition:
- For every column c, thickness is the vertical distance between the
  upper choroid boundary (as annotated) and the lower choroid boundary (CSI).
- If axial pixel resolution (um_per_px) is unknown (None), thickness is strictly reported in pixels.
- When um_per_px is provided, thickness is converted to micrometers (um):
    CT(c) = (z_lower(c) - z_upper(c)) * um_per_px [um]
- Slices are divided into equal angular sectors ("Sector 1", "Sector 2", ...).
  Anatomical labels (Temporal/Superior/Nasal/Inferior) are NOT assumed without scanner
  scan direction and eye laterality (OD/OS) metadata.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_thickness_profile(
    top: np.ndarray,
    bot: np.ndarray,
    valid: np.ndarray,
    um_per_px: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """
    Computes choroidal thickness at each column.

    Args:
        top: 1D array of upper boundary row coordinates
        bot: 1D array of lower boundary row coordinates
        valid: 1D boolean array indicating columns containing choroid
        um_per_px: Axial resolution in micrometers per pixel, or None if unknown.

    Returns:
        thickness: 1D array of thickness values (NaN for invalid columns)
        units: 'um' if um_per_px is specified, else 'px'
    """
    t = bot - top
    t[~valid] = np.nan

    if um_per_px is not None and um_per_px > 0:
        t = t * um_per_px
        units = "um"
    else:
        units = "px"

    return t, units


def compute_pixel_count_thickness(
    mask: np.ndarray,
    um_per_px: Optional[float] = None,
) -> Tuple[np.ndarray, str]:
    """
    Computes choroidal thickness by directly summing active mask pixels per column.
    Serves as an independent cross-check against boundary distance (bot - top).
    In contiguous choroid without gaps or multiple runs, boundary thickness and
    pixel-count thickness are mathematically identical.

    Args:
        mask: 2D binary mask array (H, W)
        um_per_px: Axial resolution in micrometers per pixel, or None.

    Returns:
        thickness_counts: 1D array of thickness values
        units: 'um' or 'px'
    """
    counts = mask.sum(axis=0).astype(np.float64)
    counts[counts == 0] = np.nan

    if um_per_px is not None and um_per_px > 0:
        counts = counts * um_per_px
        units = "um"
    else:
        units = "px"

    return counts, units


def summarize_thickness(
    t: np.ndarray,
    n_sectors: int = 4,
    units: str = "px",
) -> Dict[str, Any]:
    """
    Computes global and sector-wise thickness statistics across circumpapillary scan.
    Sectors are identified strictly as 'Sector 1' to 'Sector N' to avoid unverified
    anatomical quadrant assumptions.

    Args:
        t: 1D thickness profile
        n_sectors: Number of angular sectors to divide circumpapillary B-scan into (default 4)
        units: 'px' or 'um'

    Returns:
        Dictionary containing summary statistics
    """
    valid_vals = t[~np.isnan(t)]

    if valid_vals.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "sector_means": {f"sector_{i+1}": 0.0 for i in range(n_sectors)},
            "units": units,
        }

    sectors = np.array_split(t, n_sectors)
    sector_means = {
        f"sector_{i+1}": float(np.nanmean(sec)) if not np.all(np.isnan(sec)) else np.nan
        for i, sec in enumerate(sectors)
    }

    return {
        "mean": float(np.nanmean(valid_vals)),
        "std": float(np.nanstd(valid_vals)),
        "min": float(np.nanmin(valid_vals)),
        "max": float(np.nanmax(valid_vals)),
        "median": float(np.nanmedian(valid_vals)),
        "sector_means": sector_means,
        "units": units,
    }


def extract_macular_hole_center(bgr_or_path: Any) -> Tuple[Optional[int], int]:
    """
    Extracts the macular hole center column and area from an OIMHS composite scan
    (red mask: BGR [0, 0, 255] on the right half).

    Returns:
        (c_fovea, hole_area) where c_fovea is the centroid column (or None if no hole)
    """
    import cv2
    if isinstance(bgr_or_path, str):
        bgr = cv2.imread(bgr_or_path)
    else:
        bgr = bgr_or_path

    if bgr is None:
        return None, 0

    h, w_full = bgr.shape[:2]
    w_mid = w_full // 2
    right_half = bgr[:, w_mid:]

    # Red is BGR: B=0, G=0, R=255
    hole_mask = (
        (right_half[:, :, 2] == 255) &
        (right_half[:, :, 0] == 0) &
        (right_half[:, :, 1] == 0)
    )
    hole_area = int(hole_mask.sum())
    if hole_area == 0:
        return None, 0

    cols = np.where(hole_mask)[1]
    c_fovea = int(round(float(np.mean(cols))))
    return c_fovea, hole_area


def find_foveal_scans_in_dataset(scan_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Identifies the primary foveal B-scan for each eye across a cohort.
    The scan with the largest macular hole area in an eye volume represents the foveal scan.

    Returns:
        Mapping: eye_id -> {'foveal_scan': path, 'c_fovea': int, 'hole_area': int}
    """
    import os
    eye_groups: Dict[str, List[str]] = {}
    for p in scan_paths:
        eye_id = os.path.basename(os.path.dirname(p))
        eye_groups.setdefault(eye_id, []).append(p)

    fovea_map = {}
    for eye_id, paths in eye_groups.items():
        max_area = 0
        best_path = None
        best_col = None
        for p in paths:
            c_fov, area = extract_macular_hole_center(p)
            if area > max_area:
                max_area = area
                best_path = p
                best_col = c_fov
        if best_path is not None:
            fovea_map[eye_id] = {
                "foveal_scan": best_path,
                "c_fovea": best_col,
                "hole_area": max_area,
            }
    return fovea_map


def compute_foveal_thickness(
    t_pred: np.ndarray,
    t_gt: np.ndarray,
    c_fovea: int,
    window_radius: int = 25,
    units: str = "px",
) -> Dict[str, Any]:
    """
    Computes Subfoveal Choroidal Thickness (SFCT) at the foveal column c_fovea,
    and parafoveal mean thickness over a window [c_fovea - window_radius, c_fovea + window_radius].

    Replaces circumpapillary sector summaries for macular OCT scans.
    """
    W = len(t_pred)
    c = max(0, min(W - 1, int(c_fovea)))

    sfct_pred = float(t_pred[c]) if not np.isnan(t_pred[c]) else np.nan
    sfct_gt = float(t_gt[c]) if not np.isnan(t_gt[c]) else np.nan
    sfct_error = float(abs(sfct_pred - sfct_gt)) if (not np.isnan(sfct_pred) and not np.isnan(sfct_gt)) else np.nan

    c_start = max(0, c - window_radius)
    c_end = min(W, c + window_radius + 1)

    win_pred = t_pred[c_start:c_end]
    win_gt = t_gt[c_start:c_end]

    parafoveal_pred = float(np.nanmean(win_pred)) if not np.all(np.isnan(win_pred)) else np.nan
    parafoveal_gt = float(np.nanmean(win_gt)) if not np.all(np.isnan(win_gt)) else np.nan
    parafoveal_error = float(abs(parafoveal_pred - parafoveal_gt)) if (not np.isnan(parafoveal_pred) and not np.isnan(parafoveal_gt)) else np.nan

    return {
        "c_fovea": c,
        "window_radius_cols": window_radius,
        "subfoveal_thickness_pred": sfct_pred,
        "subfoveal_thickness_gt": sfct_gt,
        "subfoveal_thickness_mae": sfct_error,
        "parafoveal_thickness_pred": parafoveal_pred,
        "parafoveal_thickness_gt": parafoveal_gt,
        "parafoveal_thickness_mae": parafoveal_error,
        "units": units,
    }
