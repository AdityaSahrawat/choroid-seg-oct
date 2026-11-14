"""
Post-processing and anatomical boundary extraction for choroid segmentation.

Steps:
1. Resize probability map back to native B-scan dimensions (orig_h, orig_w).
2. Threshold to binary mask.
3. Largest connected component extraction to remove spurious isolated predictions.
4. Binary hole filling.
5. Column-wise extraction of:
   - Upper choroid boundary (as annotated)
   - Lower choroid boundary (Choroid-Sclera Interface, CSI)
6. Savitzky-Golay boundary smoothing for predictions ONLY.
   Ground-truth boundaries are extracted strictly without smoothing and without interpolation.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from scipy.signal import savgol_filter


def clean_mask(
    prob_map: np.ndarray,
    orig_hw: Tuple[int, int],
    threshold: float = 0.5,
    keep_largest_component: bool = True,
    fill_holes: bool = False,
) -> np.ndarray:
    """
    Cleans raw probability map: resizes to original resolution, thresholds,
    optionally keeps largest connected component, and fills holes.

    Args:
        prob_map: 2D numpy array of probabilities in [0, 1]
        orig_hw: Tuple of (height, width) of the original image
        threshold: Binarization probability threshold
        keep_largest_component: If True, retains only largest connected component
        fill_holes: If True, fills internal holes

    Returns:
        Cleaned binary mask of shape (orig_h, orig_w) with values {0, 1}
    """
    orig_h, orig_w = orig_hw

    # 1. Bilinear resize back to native image dimensions
    prob_resized = cv2.resize(
        prob_map,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # 2. Thresholding
    binary = (prob_resized > threshold).astype(np.uint8)

    # 3. Optional: Keep only the largest connected component
    if keep_largest_component:
        labeled, num_features = label(binary)
        if num_features > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # Ignore background
            largest_comp_id = sizes.argmax()
            binary = (labeled == largest_comp_id).astype(np.uint8)
        elif num_features == 0:
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

    # 4. Optional: Fill internal holes
    if fill_holes:
        binary = binary_fill_holes(binary).astype(np.uint8)

    return binary


def extract_boundaries(
    mask: np.ndarray,
    smooth_window: Optional[int] = None,
    poly_order: int = 2,
    interpolate_gaps: bool = False,
    max_lumen_gap: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts upper choroid boundary (Bruch's membrane) and lower choroid boundary (CSI)
    for each column using merged-band extraction.

    Anatomical merged-band algorithm:
    1. Group column active pixels into runs.
    2. Merge consecutive runs separated by small gaps (<= max_lumen_gap, default 15 px)
       representing unpainted large vessel lumens (Haller/Sattler layers).
    3. Retain the main anatomical choroid band consistent with the midline anchor,
       discarding detached strokes separated by large gaps (> 15 px, e.g. stray strokes across macular holes).

    Args:
        mask: 2D binary array of shape (H, W)
        smooth_window: Odd integer window length for Savitzky-Golay filter, or None/0 to disable.
        poly_order: Polynomial order for Savitzky-Golay filter
        interpolate_gaps: Whether to interpolate missing columns across gaps before smoothing
        max_lumen_gap: Maximum vertical gap (in pixels) across which to merge intra-choroidal runs

    Returns:
        top_boundary: 1D array of shape (W,) with upper choroid boundary row indices
        bot_boundary: 1D array of shape (W,) with lower choroid boundary row indices
        valid: 1D boolean array of shape (W,) indicating columns containing choroid
    """
    H, W = mask.shape
    top = np.full(W, np.nan, dtype=np.float64)
    bot = np.full(W, np.nan, dtype=np.float64)

    # Global choroid vertical midline anchor
    active_rows = np.flatnonzero(mask.sum(axis=1) > 0)
    if active_rows.size == 0:
        return top, bot, np.zeros(W, dtype=bool)
    anchor_mid = float(np.median(active_rows))

    # Continuity-aware merged-band extraction per column
    prev_mid = anchor_mid
    for c in range(W):
        rows = np.flatnonzero(mask[:, c])
        if rows.size > 0:
            splits = np.where(np.diff(rows) > 1)[0]
            if len(splits) == 0:
                # Exactly one contiguous run
                top[c] = float(rows[0])
                bot[c] = float(rows[-1])
            else:
                # Multiple runs: identify run coordinates
                run_starts = np.insert(rows[splits + 1], 0, rows[0])
                run_ends = np.append(rows[splits], rows[-1])
                run_lengths = run_ends - run_starts + 1

                # Group consecutive runs separated by internal lumen gaps (<= max_lumen_gap)
                merged_groups = []
                cur_group = [0]
                for i in range(len(splits)):
                    gap = run_starts[i + 1] - run_ends[i] - 1
                    if gap <= max_lumen_gap:
                        cur_group.append(i + 1)
                    else:
                        merged_groups.append(cur_group)
                        cur_group = [i + 1]
                merged_groups.append(cur_group)

                # Form merged candidates
                group_starts = [run_starts[grp[0]] for grp in merged_groups]
                group_ends = [run_ends[grp[-1]] for grp in merged_groups]
                group_mids = [(s + e) / 2.0 for s, e in zip(group_starts, group_ends)]
                group_weights = [sum(run_lengths[idx] for idx in grp) for grp in merged_groups]

                # Select group consistent with local choroid midline
                dists = [abs(m - prev_mid) for m in group_mids]
                close_indices = [idx for idx, d in enumerate(dists) if d < 40.0]
                if close_indices:
                    best_grp_idx = close_indices[int(np.argmax([group_weights[idx] for idx in close_indices]))]
                else:
                    best_grp_idx = int(np.argmin(dists))

                top[c] = float(group_starts[best_grp_idx])
                bot[c] = float(group_ends[best_grp_idx])

            prev_mid = 0.8 * prev_mid + 0.2 * ((top[c] + bot[c]) / 2.0)

    valid = ~np.isnan(top)

    if not np.any(valid):
        return top, bot, valid

    # Interpolation across gaps (only if enabled for predictions)
    if interpolate_gaps:
        valid_cols = np.flatnonzero(valid)
        missing_cols = np.flatnonzero(~valid)
        if missing_cols.size > 0 and valid_cols.size > 1:
            top_interp = top.copy()
            bot_interp = bot.copy()
            top_interp[missing_cols] = np.interp(missing_cols, valid_cols, top[valid_cols])
            bot_interp[missing_cols] = np.interp(missing_cols, valid_cols, bot[valid_cols])
            top, bot = top_interp, bot_interp

    # Smoothing (predictions ONLY; skipped for ground truth)
    if smooth_window is not None and smooth_window > 0 and W >= smooth_window:
        window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        if window > W:
            window = W if W % 2 == 1 else W - 1

        if window > poly_order:
            top = savgol_filter(top, window_length=window, polyorder=poly_order)
            bot = savgol_filter(bot, window_length=window, polyorder=poly_order)

    # Re-mask non-valid columns with NaN
    top[~valid] = np.nan
    bot[~valid] = np.nan

    return top, bot, valid


def extract_gt_boundaries(
    mask: np.ndarray,
    max_lumen_gap: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts raw ground-truth boundaries without smoothing and without interpolation,
    using merged-band extraction across internal vessel lumens.
    """
    return extract_boundaries(mask, smooth_window=None, interpolate_gaps=False, max_lumen_gap=max_lumen_gap)


def extract_pred_boundaries(
    mask: np.ndarray,
    smooth_window: Optional[int] = None,
    poly_order: int = 2,
    interpolate_gaps: Optional[bool] = None,
    max_lumen_gap: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts predicted boundaries with optional gap interpolation and Savitzky-Golay smoothing,
    using merged-band extraction across internal vessel lumens.
    """
    if interpolate_gaps is None:
        interpolate_gaps = bool(smooth_window is not None and smooth_window > 0)
    return extract_boundaries(
        mask,
        smooth_window=smooth_window,
        poly_order=poly_order,
        interpolate_gaps=interpolate_gaps,
        max_lumen_gap=max_lumen_gap,
    )
