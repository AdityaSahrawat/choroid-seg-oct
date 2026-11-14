"""
Multi-Seed Experiment Runner and Results Aggregator.

Runs a configuration across multiple random seeds (e.g. 42, 43, 44),
evaluates each on the test split, and computes the final benchmark table
reporting Mean ± Std across runs.
"""

import argparse
import copy
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List
import numpy as np
import pandas as pd
import yaml

from src.train import train
from src.test_eval import main as run_test_eval


def aggregate_results(seed_summary_paths: List[str], out_dir: str) -> pd.DataFrame:
    records = []
    for path in seed_summary_paths:
        with open(path, "r") as f:
            data = json.load(f)
        row = {
            "dice": data.get("mean_dice"),
            "iou": data.get("mean_iou"),
            "hd95_px": data.get("mean_hd95_px"),
            "upper_mae_px": data.get("mean_upper_mae_px"),
            "lower_mae_px": data.get("mean_lower_mae_px"),
            "thickness_mae": data.get("mean_thickness_mae"),
            "cohort_pearson_r": data.get("cohort_pearson_r_per_scan"),
        }
        ba = data.get("bland_altman_per_scan", {})
        if ba:
            row["ba_bias"] = ba.get("bias")
            row["ba_loa_low"] = ba.get("loa_low")
            row["ba_loa_high"] = ba.get("loa_high")
        records.append(row)

    df = pd.DataFrame(records)

    summary_rows = []
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        summary_rows.append({
            "metric": col,
            "mean": mean_val,
            "std": std_val,
            "formatted": f"{mean_val:.4f} ± {std_val:.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, "multi_seed_aggregated.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n=======================================================")
    print("        MULTI-SEED BENCHMARK TABLE (Mean ± Std)       ")
    print("=======================================================")
    for _, r in summary_df.iterrows():
        print(f"  {r['metric']:<20}: {r['formatted']}")
    print("=======================================================")
    print(f"Saved aggregated table to: {csv_path}\n")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Run experiments over multiple seeds and aggregate results.")
    parser.add_argument("--config", type=str, required=True, help="Base YAML configuration file.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="List of random seeds (default: 42 43 44)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    base_name = base_cfg.get("name", "experiment")
    seed_summary_paths = []

    for seed in args.seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["name"] = f"{base_name}_seed{seed}"
        cfg["seed"] = seed

        print(f"\n>>> Running Seed {seed} for {cfg['name']} <<<")
        train(cfg)

        # Evaluate best model
        ckpt_path = os.path.join(cfg.get("output_dir", "outputs"), cfg["name"], "best_model.pt")
        test_split = cfg.get("data", {}).get("test_split", "data/splits/test.txt")
        test_out_dir = os.path.join(cfg.get("output_dir", "outputs"), cfg["name"], "test_eval")

        if os.path.isfile(ckpt_path) and os.path.isfile(test_split):
            # Run test evaluation
            import sys
            saved_argv = sys.argv
            sys.argv = [
                "test_eval.py",
                "--checkpoint", ckpt_path,
                "--split_file", test_split,
                "--output_dir", test_out_dir,
            ]
            try:
                run_test_eval()
                summary_path = os.path.join(test_out_dir, "test_summary.json")
                if os.path.isfile(summary_path):
                    seed_summary_paths.append(summary_path)
            finally:
                sys.argv = saved_argv

    if seed_summary_paths:
        agg_dir = os.path.join(base_cfg.get("output_dir", "outputs"), base_name)
        aggregate_results(seed_summary_paths, agg_dir)
    else:
        print("\nNote: Evaluation skipped or no checkpoints found (ensure data/splits/test.txt exists).")


if __name__ == "__main__":
    main()
