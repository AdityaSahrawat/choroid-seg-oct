"""
Unit tests for Choroid OCT Segmentation Pipeline.
"""

import numpy as np
import pytest
import torch

from src.dataset import compute_boundary_weight_map
from src.evaluate import (
    compute_bland_altman,
    compute_boundary_errors,
    compute_dice_iou,
    compute_hd95,
    compute_thickness_metrics,
)
from src.losses import BCEDiceLoss, DiceLoss
from src.models import build_model
from src.postprocess import (
    clean_mask,
    extract_boundaries,
    extract_gt_boundaries,
    extract_pred_boundaries,
)
from src.preprocess import OCTPreprocessor
from src.thickness import compute_thickness_profile, summarize_thickness


def test_preprocessor():
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    preproc = OCTPreprocessor(use_clahe=True, denoise_method="median", median_kernel=3)
    out = preproc(img)
    assert out.shape == (100, 100)
    assert out.dtype == np.uint8


def test_boundary_weight_map():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 10:50] = 1
    w = compute_boundary_weight_map(mask, w0=5.0, sigma=3.0)
    assert w.shape == (64, 64)
    assert w.max() > 1.0  # Boundary contours get higher weights
    assert w.min() >= 1.0


def test_models_forward():
    x = torch.randn(2, 1, 128, 128)

    # Scratch U-Net
    m1 = build_model("unet", encoder_name="resnet34", encoder_weights=None)
    y1 = m1(x)
    assert y1.shape == (2, 1, 128, 128)

    # U-Net++ with scSE attention
    m2 = build_model("unetplusplus", encoder_name="resnet34", encoder_weights=None, decoder_attention_type="scse")
    y2 = m2(x)
    assert y2.shape == (2, 1, 128, 128)


def test_losses():
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    weights = torch.ones_like(targets) * 2.0

    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    loss_unweighted = criterion(logits, targets)
    loss_weighted = criterion(logits, targets, weights)

    assert not torch.isnan(loss_unweighted)
    assert not torch.isnan(loss_weighted)
    assert loss_unweighted > 0
    assert loss_weighted > 0


def test_postprocess_and_boundaries():
    H, W = 100, 200
    prob_map = np.zeros((H, W), dtype=np.float32)
    # Choroid between rows 30 and 60
    prob_map[30:60, 20:180] = 0.9

    cleaned = clean_mask(prob_map, orig_hw=(H, W), threshold=0.5)
    assert cleaned.shape == (H, W)
    assert cleaned.dtype == np.uint8
    assert cleaned[45, 50] == 1
    assert cleaned[10, 10] == 0

    # Explicit test: --no_postprocess raw path STILL resizes to native resolution (e.g. 800x1100)
    prob_network_size = np.zeros((512, 512), dtype=np.float32)
    raw_unprocessed = clean_mask(
        prob_network_size,
        orig_hw=(800, 1100),
        threshold=0.5,
        keep_largest_component=False,
        fill_holes=False,
    )
    assert raw_unprocessed.shape == (800, 1100), "Raw path must strictly restore native scan resolution!"

    # Ground truth: extracted strictly raw without smoothing or gap interpolation
    top_gt, bot_gt, valid_gt = extract_gt_boundaries(cleaned)
    assert len(top_gt) == W
    assert top_gt[50] == 30.0
    assert bot_gt[50] == 59.0
    assert np.isnan(top_gt[5])  # Gap column remains NaN

    # Prediction: extracted with SavGol smoothing
    top_p, bot_p, valid_p = extract_pred_boundaries(cleaned, smooth_window=15, poly_order=2)
    assert len(top_p) == W
    assert valid_p[50] == True
    assert top_p[50] == pytest.approx(30, abs=2)


def test_thickness_null_and_um():
    top = np.array([20.0, 20.0, np.nan])
    bot = np.array([50.0, 60.0, np.nan])
    valid = np.array([True, True, False])

    # Case 1: um_per_px is None (reporting strictly in pixels)
    t_px, units_px = compute_thickness_profile(top, bot, valid, um_per_px=None)
    assert units_px == "px"
    assert t_px[0] == 30.0
    assert t_px[1] == 40.0
    assert np.isnan(t_px[2])

    summary_px = summarize_thickness(t_px, n_sectors=2, units=units_px)
    assert summary_px["units"] == "px"
    assert summary_px["mean"] == 35.0
    assert "sector_1" in summary_px["sector_means"]
    assert "sector_2" in summary_px["sector_means"]

    # Case 2: um_per_px is provided (e.g., 3.87 um/px)
    t_um, units_um = compute_thickness_profile(top, bot, valid, um_per_px=3.87)
    assert units_um == "um"
    assert t_um[0] == pytest.approx(30.0 * 3.87)
    assert t_um[1] == pytest.approx(40.0 * 3.87)

    summary_um = summarize_thickness(t_um, n_sectors=2, units=units_um)
    assert summary_um["units"] == "um"
    assert summary_um["mean"] == pytest.approx(35.0 * 3.87)


def test_evaluation_metrics():
    mask1 = np.zeros((50, 50), dtype=np.uint8)
    mask2 = np.zeros((50, 50), dtype=np.uint8)
    mask1[10:30, 10:30] = 1
    mask2[10:30, 10:30] = 1

    dice, iou = compute_dice_iou(mask1, mask2)
    assert dice == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)

    hd95 = compute_hd95(mask1, mask2)
    assert hd95 == pytest.approx(0.0, abs=1e-3)

    t1 = np.array([10.0, 20.0, 30.0])
    t2 = np.array([11.0, 19.0, 32.0])
    t_metrics = compute_thickness_metrics(t1, t2)
    assert t_metrics["thickness_mae"] > 0
    assert -1.0 <= t_metrics["pearson_r"] <= 1.0

    # Per-scan Bland-Altman
    p_scans = np.array([100.0, 120.0, 150.0])
    g_scans = np.array([102.0, 118.0, 153.0])
    ba = compute_bland_altman(p_scans, g_scans)
    assert not np.isnan(ba["bias"])
    assert not np.isnan(ba["sd"])
    assert ba["loa_low"] < ba["bias"] < ba["loa_high"]


def test_gt_boundary_extraction_raw_and_nans_preserved():
    """
    Test 1 (Requested): Verify ground-truth boundary extraction returns exact raw values
    with zero Savitzky-Golay smoothing and strictly preserves NaNs across gap columns.
    """
    H, W = 100, 50
    mask = np.zeros((H, W), dtype=np.uint8)

    # Intentionally create step-like discrete edges on columns 10..39
    # Col 10..19: rows 30..60
    # Col 20..29: rows 40..70 (sudden step)
    # Col 30..39: rows 35..65
    # Col 0..9 and 40..49: completely empty (NaN gaps)
    mask[30:61, 10:20] = 1
    mask[40:71, 20:30] = 1
    mask[35:66, 30:40] = 1

    top_gt, bot_gt, valid_gt = extract_gt_boundaries(mask)

    # 1. Gaps must remain strictly NaN (no gap interpolation!)
    assert np.all(np.isnan(top_gt[0:10]))
    assert np.all(np.isnan(bot_gt[0:10]))
    assert np.all(np.isnan(top_gt[40:50]))
    assert np.all(np.isnan(bot_gt[40:50]))
    assert np.all(~valid_gt[0:10])
    assert np.all(~valid_gt[40:50])

    # 2. Valid regions must match exact raw row indices (no smoothing!)
    assert np.all(top_gt[10:20] == 30.0)
    assert np.all(bot_gt[10:20] == 60.0)
    assert np.all(top_gt[20:30] == 40.0)
    assert np.all(bot_gt[20:30] == 70.0)
    assert np.all(top_gt[30:40] == 35.0)
    assert np.all(bot_gt[30:40] == 65.0)


def test_boundary_weight_map_alignment_with_augmented_mask():
    """
    Test 2 (Requested): Verify boundary weight map lines up with the augmented mask.
    When a shift/translation is applied, the high weights must sit directly on the new edge.
    """
    H, W = 128, 128
    mask_orig = np.zeros((H, W), dtype=np.uint8)
    mask_orig[30:70, 30:70] = 1

    # Apply a known shift: shift right by 20 pixels and down by 15 pixels
    shift_r, shift_c = 15, 20
    mask_shifted = np.zeros((H, W), dtype=np.uint8)
    mask_shifted[30 + shift_r : 70 + shift_r, 30 + shift_c : 70 + shift_c] = 1

    # Compute boundary weights directly on shifted mask
    w = compute_boundary_weight_map(mask_shifted, w0=5.0, sigma=3.0)

    # The contour edges of mask_shifted are:
    # top edge at row (30 + shift_r) = 45
    # bottom edge at row (70 + shift_r - 1) = 84
    # left edge at col (30 + shift_c) = 50
    # right edge at col (70 + shift_c - 1) = 89
    edge_row_top = 30 + shift_r
    edge_col_left = 30 + shift_c

    # On the shifted edge, weights must be at their maximum (> 4.5, w0=5.0)
    assert w[edge_row_top, 60] > 4.5
    assert w[edge_row_top + 5, edge_col_left] > 4.5

    # Far away from the shifted edge (e.g. at the original unshifted position row 30, col 30),
    # weight must be near baseline 1.0!
    assert w[30, 30] == pytest.approx(1.0, abs=0.1)


def test_per_scan_bland_altman_one_point_per_scan():
    """
    Test 3 (Requested): Verify per-scan Bland-Altman computes agreement on exactly
    one point per scan (independent subject means) rather than pooled columns.
    """
    # 15 distinct test scans with known mean thicknesses
    N = 15
    gt_means = np.linspace(150.0, 350.0, N)
    # Systematic bias of +4.0 pixels with small random deviations
    pred_means = gt_means + 4.0 + np.array([-1, 0, 1, -2, 2, -1, 1, 0, 1, -1, 2, -2, 0, 1, -1]) * 0.5

    ba = compute_bland_altman(pred_means, gt_means)

    expected_diffs = pred_means - gt_means
    assert len(expected_diffs) == N
    assert ba["bias"] == pytest.approx(float(np.mean(expected_diffs)), abs=1e-5)
    assert ba["sd"] == pytest.approx(float(np.std(expected_diffs, ddof=1)), abs=1e-5)
    assert ba["loa_high"] == pytest.approx(ba["bias"] + 1.96 * ba["sd"], abs=1e-5)
    assert ba["loa_low"] == pytest.approx(ba["bias"] - 1.96 * ba["sd"], abs=1e-5)


def test_merged_band_extraction_merges_lumens_and_discards_stray_strokes():
    """
    Test 4 (New): Verify merged-band extraction merges intra-choroidal vessel lumen gaps
    (<= 15 px) while safely discarding detached external strokes (> 15 px).
    """
    H, W = 120, 50
    mask = np.zeros((H, W), dtype=np.uint8)

    # Base choroid layer from row 40 to 90 across columns 10 to 40
    mask[40:91, 10:40] = 1

    # Col 20: simulate an unpainted large vessel lumen (Haller layer void) at rows 55..62 (gap = 8 px)
    mask[55:63, 20] = 0

    # Col 25: simulate a detached stray annotation stroke high up at rows 10..15 (gap to choroid = 25 px > 15 px)
    mask[10:16, 25] = 1

    top, bot, valid = extract_gt_boundaries(mask, max_lumen_gap=15)

    assert valid[20]
    assert valid[25]

    # At Col 20: lumen gap of 8 px is <= 15 px, so upper run (40..54) and lower run (63..90)
    # must be merged into the complete choroid band!
    assert top[20] == 40.0
    assert bot[20] == 90.0
    assert (bot[20] - top[20]) == 50.0

    # At Col 25: stray stroke at rows 10..15 has gap of 25 px (> 15 px) and is far from anchor.
    # It must be discarded, retaining the true choroid band (40..90)!
    assert top[25] == 40.0
    assert bot[25] == 90.0


def test_curve_ae95_and_hd95_metrics():
    """
    Test 5 (New): Verify standardized curve AE95 (vertical error) and HD95 (2D Euclidean).
    """
    from src.evaluate import compute_curve_ae95, compute_curve_hd95

    W = 100
    gt = np.full(W, 50.0)
    pred = np.full(W, 50.0)

    # 89 columns match perfectly, 5 columns have +2.0 px vertical error, 6 columns have +10.0 px error
    pred[50:55] = 52.0
    pred[94:100] = 60.0

    ae95 = compute_curve_ae95(pred, gt)
    # 95th percentile of errors: 89 zeros, 5 twos, 6 tens -> at index 94.05, value is 10.0
    assert ae95 == pytest.approx(10.0, abs=0.1)

    hd95 = compute_curve_hd95(pred, gt)
    assert not np.isnan(hd95)
    assert hd95 > 0.0

