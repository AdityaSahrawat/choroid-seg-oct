"""
Automated comparison and reporting script for Choroid OCT segmentation:
Baseline (unet_r34) vs. Improved (unetpp_r34_scse with scSE attention and boundary loss).

Generates:
1. Head-to-head comparison table across key metrics.
2. Per-eye Subfoveal Choroidal Thickness (SFCT) comparison across the 19 foveal test eyes.
3. Quality robustness comparison (Good vs. Poor quality scans).
4. Formatted Markdown summary.
"""

import os
import json
import numpy as np
import pandas as pd

def load_summary(summary_path: str):
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r") as f:
        return json.load(f)

def generate_comparison(
    baseline_dir: str = "outputs/unet_r34/test_eval",
    improved_dir: str = "outputs/unetpp_r34_scse/test_eval",
    out_dir: str = "outputs/comparison"
):
    os.makedirs(out_dir, exist_ok=True)
    base_sum = load_summary(os.path.join(baseline_dir, "test_summary.json"))
    imp_sum = load_summary(os.path.join(improved_dir, "test_summary.json"))

    base_csv = os.path.join(baseline_dir, "test_metrics_per_scan.csv")
    imp_csv = os.path.join(improved_dir, "test_metrics_per_scan.csv")
    
    df_base = pd.read_csv(base_csv) if os.path.exists(base_csv) else None
    df_imp = pd.read_csv(imp_csv) if os.path.exists(imp_csv) else None

    um_per_px = 3.75 # Nominal assumption

    print("\n" + "="*80)
    print("HEAD-TO-HEAD COMPARISON: BASELINE vs. IMPROVED (CHOROID SEGMENTATION)")
    print("="*80)

    rows = []
    if base_sum and imp_sum:
        metrics = [
            ("Dice Similarity (DSC)", "mean_dice", True, ""),
            ("Intersection over Union (IoU)", "mean_iou", True, ""),
            ("Bruch's Membrane MAE (px)", "mean_bruchs_membrane_mae_px", False, "px"),
            ("CSI (Lower Boundary) MAE (px)", "mean_csi_mae_px", False, "px"),
            ("Choroidal Thickness MAE (px)", "mean_thickness_mae", False, "px"),
            ("2D Mask Contour HD95 (px)", "mean_hd95_mask_px", False, "px"),
            ("Bruch's Membrane HD95 (px)", "mean_hd95_bruchs_membrane_px", False, "px"),
            ("CSI Lower Boundary HD95 (px)", "mean_hd95_csi_px", False, "px"),
            ("Bruch's Membrane AE95 (px)", "mean_ae95_bruchs_membrane_px", False, "px"),
            ("CSI Lower Boundary AE95 (px)", "mean_ae95_csi_px", False, "px"),
            ("Per-Scan Pearson r", "cohort_pearson_r_per_scan", True, ""),
        ]

        for label, key, higher_better, unit in metrics:
            val_base = base_sum.get(key, np.nan)
            val_imp = imp_sum.get(key, np.nan)
            if not np.isnan(val_base) and not np.isnan(val_imp):
                diff = val_imp - val_base
                rel_diff = (diff / abs(val_base)) * 100.0 if val_base != 0 else 0.0
                better = (diff > 0) if higher_better else (diff < 0)
                mark = "(Improved)" if better else "(Baseline better)" if diff != 0 else "(Equal)"
                rows.append({
                    "Metric": label,
                    "Baseline (U-Net R34)": f"{val_base:.4f}" if "px" not in label and "MAE" not in label else f"{val_base:.2f}",
                    "Improved (U-Net++ scSE)": f"{val_imp:.4f}" if "px" not in label and "MAE" not in label else f"{val_imp:.2f}",
                    "Delta": f"{diff:+.2f} ({rel_diff:+.1f}%) {mark}"
                })

        # Bland-Altman
        ba_b = base_sum.get("bland_altman_per_scan", {})
        ba_i = imp_sum.get("bland_altman_per_scan", {})
        rows.append({
            "Metric": "Bland-Altman Mean Bias (px)",
            "Baseline (U-Net R34)": f"{ba_b.get('bias', np.nan):.3f}",
            "Improved (U-Net++ scSE)": f"{ba_i.get('bias', np.nan):.3f}",
            "Delta": f"{ba_i.get('bias', np.nan) - ba_b.get('bias', np.nan):+.3f} px"
        })
        rows.append({
            "Metric": "Bland-Altman 95% LoA (px)",
            "Baseline (U-Net R34)": f"[{ba_b.get('loa_low', np.nan):.2f}, {ba_b.get('loa_high', np.nan):.2f}]",
            "Improved (U-Net++ scSE)": f"[{ba_i.get('loa_low', np.nan):.2f}, {ba_i.get('loa_high', np.nan):.2f}]",
            "Delta": f"Span: {ba_b.get('loa_high',0)-ba_b.get('loa_low',0):.2f} -> {ba_i.get('loa_high',0)-ba_i.get('loa_low',0):.2f} px"
        })

        # Foveal Summary
        fov_b = base_sum.get("foveal_summary", {})
        fov_i = imp_sum.get("foveal_summary", {})
        rows.append({
            "Metric": "Subfoveal Thickness MAE (px)",
            "Baseline (U-Net R34)": f"{fov_b.get('mean_subfoveal_mae', np.nan):.2f} px ({fov_b.get('mean_subfoveal_mae', np.nan)*um_per_px:.1f} um)",
            "Improved (U-Net++ scSE)": f"{fov_i.get('mean_subfoveal_mae', np.nan):.2f} px ({fov_i.get('mean_subfoveal_mae', np.nan)*um_per_px:.1f} um)",
            "Delta": f"{fov_i.get('mean_subfoveal_mae', np.nan) - fov_b.get('mean_subfoveal_mae', np.nan):+.2f} px"
        })
        rows.append({
            "Metric": "Parafoveal Thickness MAE (px)",
            "Baseline (U-Net R34)": f"{fov_b.get('mean_parafoveal_mae', np.nan):.2f} px ({fov_b.get('mean_parafoveal_mae', np.nan)*um_per_px:.1f} um)",
            "Improved (U-Net++ scSE)": f"{fov_i.get('mean_parafoveal_mae', np.nan):.2f} px ({fov_i.get('mean_parafoveal_mae', np.nan)*um_per_px:.1f} um)",
            "Delta": f"{fov_i.get('mean_parafoveal_mae', np.nan) - fov_b.get('mean_parafoveal_mae', np.nan):+.2f} px"
        })

    df_comp = pd.DataFrame(rows)
    print(df_comp.to_string(index=False))

    # Per-eye SFCT Table across 19 foveal eyes
    if df_base is not None and df_imp is not None:
        fov_base = df_base[df_base['subfoveal_thickness_gt'].notna()].copy()
        fov_imp = df_imp[df_imp['subfoveal_thickness_gt'].notna()].copy()

        merged_fov = pd.merge(
            fov_base[['eye_id', 'sample', 'c_fovea', 'subfoveal_thickness_gt', 'subfoveal_thickness_pred', 'subfoveal_thickness_mae']],
            fov_imp[['eye_id', 'sample', 'subfoveal_thickness_pred', 'subfoveal_thickness_mae']],
            on=['eye_id', 'sample'],
            suffixes=('_base', '_imp')
        )

        merged_fov['sfct_gt_um'] = merged_fov['subfoveal_thickness_gt'] * um_per_px
        merged_fov['sfct_base_um'] = merged_fov['subfoveal_thickness_pred_base'] * um_per_px
        merged_fov['sfct_imp_um'] = merged_fov['subfoveal_thickness_pred_imp'] * um_per_px

        print("\n" + "="*80)
        print("PER-EYE SUBFOVEAL CHOROIDAL THICKNESS (SFCT) ACROSS 19 FOVEAL TEST EYES")
        print("="*80)
        print(merged_fov[['eye_id', 'sample', 'c_fovea', 'subfoveal_thickness_gt', 'subfoveal_thickness_pred_base', 'subfoveal_thickness_mae_base', 'subfoveal_thickness_pred_imp', 'subfoveal_thickness_mae_imp']].to_string(index=False))

        merged_fov.to_csv(os.path.join(out_dir, "sfct_19_eyes_comparison.csv"), index=False)

    df_comp.to_csv(os.path.join(out_dir, "model_comparison_table.csv"), index=False)
    with open(os.path.join(out_dir, "comparison_summary.json"), "w") as f:
        json.dump({"baseline": base_sum, "improved": imp_sum}, f, indent=2)

if __name__ == "__main__":
    generate_comparison()

