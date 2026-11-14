"""
Detailed empirical audit of Yellow (True Choroid) layer across OIMHS test set.

Evaluates:
1. First-to-last row thickness
2. Main-run thickness
3. Pixel count thickness
4. Merged-band thickness (merge gaps <= 15 px)
5. Multi-run and gap characteristics for Yellow
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd

def analyze_yellow_scan(bgr_path: str):
    bgr = cv2.imread(bgr_path)
    if bgr is None:
        return None
    w_mid = bgr.shape[1] // 2
    raw = bgr[:, :w_mid]
    right = bgr[:, w_mid:]
    
    # Yellow is BGR [0, 255, 255] (Choroid)
    choroid_yellow = ((right[:, :, 0] == 0) & (right[:, :, 1] == 255) & (right[:, :, 2] == 255)).astype(np.uint8)
    # Green is BGR [0, 255, 0] (Retina)
    retina_green = ((right[:, :, 1] == 255) & (right[:, :, 0] == 0) & (right[:, :, 2] == 0)).astype(np.uint8)
    # Red is BGR [0, 0, 255] (Macular Hole)
    hole_red = ((right[:, :, 2] == 255) & (right[:, :, 0] == 0) & (right[:, :, 1] == 0)).astype(np.uint8)
    
    H, W = choroid_yellow.shape
    col_records = []
    gap_records = []
    
    active_rows = np.flatnonzero(choroid_yellow.sum(axis=1) > 0)
    if active_rows.size == 0:
        return None
    anchor_mid = float(np.median(active_rows))
    prev_mid = anchor_mid
    
    for c in range(W):
        rows = np.flatnonzero(choroid_yellow[:, c])
        if rows.size == 0:
            continue
            
        splits = np.where(np.diff(rows) > 1)[0]
        n_runs = len(splits) + 1
        
        t_first_last = float(rows[-1] - rows[0] + 1)
        t_pixel_count = float(rows.size)
        
        run_starts = np.insert(rows[splits + 1], 0, rows[0])
        run_ends = np.append(rows[splits], rows[-1])
        run_lengths = run_ends - run_starts + 1
        mids = (run_starts + run_ends) / 2.0
        
        # Main-run selection
        dists = np.abs(mids - prev_mid)
        close_mask = dists < 30.0
        if np.any(close_mask):
            best_idx = np.where(close_mask)[0][np.argmax(run_lengths[close_mask])]
        else:
            best_idx = int(np.argmin(dists))
        t_main_run = float(run_ends[best_idx] - run_starts[best_idx] + 1)
        prev_mid = 0.8 * prev_mid + 0.2 * mids[best_idx]
        
        # Gaps analysis
        if n_runs > 1:
            for i in range(len(splits)):
                gap_start = run_ends[i] + 1
                gap_end = run_starts[i + 1] - 1
                gap_size = gap_end - gap_start + 1
                gap_slice_raw = raw[gap_start:gap_end + 1, c]
                mean_intensity = float(np.mean(gap_slice_raw)) if gap_slice_raw.size > 0 else 0.0
                gap_records.append({
                    'gap_size': gap_size,
                    'mean_intensity': mean_intensity,
                    'gap_start': gap_start,
                    'gap_end': gap_end
                })
                
        # Merged-band
        merged_groups = []
        cur_group = [0]
        for i in range(len(splits)):
            gap = run_starts[i+1] - run_ends[i] - 1
            if gap <= 15:
                cur_group.append(i + 1)
            else:
                merged_groups.append(cur_group)
                cur_group = [i + 1]
        merged_groups.append(cur_group)
        group_pixels = [sum(run_lengths[idx] for idx in grp) for grp in merged_groups]
        best_grp = merged_groups[np.argmax(group_pixels)]
        t_merged = float(run_ends[best_grp[-1]] - run_starts[best_grp[0]] + 1)
        
        col_records.append({
            'n_runs': n_runs,
            't_first_last': t_first_last,
            't_pixel_count': t_pixel_count,
            't_main_run': t_main_run,
            't_merged': t_merged
        })
        
    return col_records, gap_records

def main():
    with open('data/splits/test.txt', 'r') as f:
        test_lines = [line.strip().split(',')[0] for line in f if line.strip()]
        
    print(f"Auditing Yellow (True Choroid) on {len(test_lines)} test scans...")
    all_col_records = []
    all_gap_records = []
    
    for idx, p in enumerate(test_lines):
        res = analyze_yellow_scan(p)
        if res is not None:
            cols, gaps = res
            all_col_records.extend(cols)
            all_gap_records.extend(gaps)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(test_lines)} scans...")
            
    df_cols = pd.DataFrame(all_col_records)
    df_gaps = pd.DataFrame(all_gap_records)
    
    um_per_px = 3.75
    
    print("\n" + "="*70)
    print("TRUE CHOROID (YELLOW BGR [0, 255, 255]) THICKNESS DISTRIBUTION")
    print("="*70)
    print(f"Total choroid columns analyzed: {len(df_cols)}")
    print(f"Columns with 1 run:  {(df_cols['n_runs'] == 1).sum()} ({(df_cols['n_runs'] == 1).mean()*100:.2f}%)")
    print(f"Columns with >=2 runs: {(df_cols['n_runs'] >= 2).sum()} ({(df_cols['n_runs'] >= 2).mean()*100:.2f}%)")
    
    metrics = ['t_first_last', 't_main_run', 't_pixel_count', 't_merged']
    names = [
        '1. First-to-last row (span)',
        '2. Main run (current postprocess)',
        '3. Pixel count (sum active px)',
        '4. Merged band (merge gap <= 15px)'
    ]
    
    print("\n--- Summary Statistics (Pixels) ---")
    print(f"{'Definition':<38} | {'Median':<8} | {'IQR':<14} | {'Mean':<8} | {'Std':<8}")
    print("-" * 85)
    for m, name in zip(metrics, names):
        med = df_cols[m].median()
        q25 = df_cols[m].quantile(0.25)
        q75 = df_cols[m].quantile(0.75)
        iqr = q75 - q25
        mean = df_cols[m].mean()
        std = df_cols[m].std()
        print(f"{name:<38} | {med:<8.2f} | [{q25:.1f}, {q75:.1f}] ({iqr:.1f}) | {mean:<8.2f} | {std:<8.2f}")

    print("\n--- Summary Statistics (Nominal Micrometers @ 3.75 um/px) ---")
    print(f"{'Definition':<38} | {'Median':<8} | {'IQR':<14} | {'Mean':<8} | {'Std':<8}")
    print("-" * 85)
    for m, name in zip(metrics, names):
        med = df_cols[m].median() * um_per_px
        q25 = df_cols[m].quantile(0.25) * um_per_px
        q75 = df_cols[m].quantile(0.75) * um_per_px
        iqr = q75 - q25
        mean = df_cols[m].mean() * um_per_px
        std = df_cols[m].std() * um_per_px
        print(f"{name:<38} | {med:<8.1f} | [{q25:.1f}, {q75:.1f}] ({iqr:.1f}) | {mean:<8.1f} | {std:<8.1f}")
        
    print("\n" + "="*70)
    print("ANALYSIS OF GAPS IN YELLOW MULTI-RUN COLUMNS")
    print("="*70)
    print(f"Total gaps identified in Yellow: {len(df_gaps)}")
    if len(df_gaps) > 0:
        print(f"Gap size Median: {df_gaps['gap_size'].median():.1f} px, IQR: [{df_gaps['gap_size'].quantile(0.25):.1f}, {df_gaps['gap_size'].quantile(0.75):.1f}]")
        print(f"Gap size Mean: {df_gaps['gap_size'].mean():.2f} px")
        print(f"Gaps <= 5 px:   {(df_gaps['gap_size'] <= 5).sum()} ({(df_gaps['gap_size'] <= 5).mean()*100:.2f}%)")
        print(f"Gaps <= 10 px:  {(df_gaps['gap_size'] <= 10).sum()} ({(df_gaps['gap_size'] <= 10).mean()*100:.2f}%)")
        print(f"Gaps <= 15 px:  {(df_gaps['gap_size'] <= 15).sum()} ({(df_gaps['gap_size'] <= 15).mean()*100:.2f}%)")
        print(f"Gaps > 15 px:   {(df_gaps['gap_size'] > 15).sum()} ({(df_gaps['gap_size'] > 15).mean()*100:.2f}%)")
        print(f"Gaps <= 15 px mean intensity: {df_gaps[df_gaps['gap_size'] <= 15]['mean_intensity'].mean():.1f} (dark vessel lumens)")

if __name__ == '__main__':
    main()

