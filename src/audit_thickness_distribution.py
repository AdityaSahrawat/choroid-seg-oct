"""
Detailed empirical audit of choroid thickness definitions and gap structure across OIMHS.

Evaluates:
1. First-to-last row thickness: bot_last - top_first
2. Main-run thickness: current extract_boundaries algorithm
3. Pixel count thickness: sum of active choroid pixels per column
4. Merged-band thickness: merge runs separated by small gaps (< 15 px)

Analyzes multi-run columns:
- Distribution of gap sizes between runs
- Location of gaps: internal (inside the choroid band, between vessel lumens) vs external (above Bruch's membrane / hole)
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd

def analyze_scan(bgr_path: str):
    bgr = cv2.imread(bgr_path)
    if bgr is None:
        return None
    w_mid = bgr.shape[1] // 2
    raw = bgr[:, :w_mid]
    right = bgr[:, w_mid:]
    
    # Choroid mask: Green (BGR: [0, 255, 0])
    choroid = ((right[:, :, 1] == 255) & (right[:, :, 0] == 0) & (right[:, :, 2] == 0)).astype(np.uint8)
    # Hole mask: Red (BGR: [0, 0, 255])
    hole = ((right[:, :, 2] == 255) & (right[:, :, 0] == 0) & (right[:, :, 1] == 0)).astype(np.uint8)
    # Retina mask: Blue (BGR: [255, 0, 0])
    retina = ((right[:, :, 0] == 255) & (right[:, :, 1] == 0) & (right[:, :, 2] == 0)).astype(np.uint8)

    H, W = choroid.shape
    
    col_records = []
    gap_records = []
    
    # Global choroid vertical anchor
    active_rows = np.flatnonzero(choroid.sum(axis=1) > 0)
    if active_rows.size == 0:
        return None
    anchor_mid = float(np.median(active_rows))
    prev_mid = anchor_mid

    for c in range(W):
        rows = np.flatnonzero(choroid[:, c])
        if rows.size == 0:
            continue
            
        splits = np.where(np.diff(rows) > 1)[0]
        n_runs = len(splits) + 1
        
        # 1. First-to-last row
        t_first_last = float(rows[-1] - rows[0] + 1) # or rows[-1] - rows[0]
        
        # 2. Pixel count
        t_pixel_count = float(rows.size)
        
        # Runs decomposition
        run_starts = np.insert(rows[splits + 1], 0, rows[0])
        run_ends = np.append(rows[splits], rows[-1])
        run_lengths = run_ends - run_starts + 1
        mids = (run_starts + run_ends) / 2.0
        
        # 3. Main-run selection (current logic)
        dists = np.abs(mids - prev_mid)
        close_mask = dists < 30.0
        if np.any(close_mask):
            best_idx = np.where(close_mask)[0][np.argmax(run_lengths[close_mask])]
        else:
            best_idx = int(np.argmin(dists))
        t_main_run = float(run_ends[best_idx] - run_starts[best_idx] + 1)
        prev_mid = 0.8 * prev_mid + 0.2 * mids[best_idx]
        
        # 4. Merged-band selection
        # Identify gaps between consecutive runs
        if n_runs > 1:
            for i in range(len(splits)):
                gap_start = run_ends[i] + 1
                gap_end = run_starts[i + 1] - 1
                gap_size = gap_end - gap_start + 1
                
                # Check what is inside the gap
                gap_slice_hole = hole[gap_start:gap_end + 1, c]
                gap_slice_retina = retina[gap_start:gap_end + 1, c]
                gap_slice_raw = raw[gap_start:gap_end + 1, c]
                
                # Internal vs external:
                # Is the upper run sitting inside the macular hole or high up in retina?
                # If gap_size < 15 and no hole pixels, it's an internal vessel lumen inside choroid
                has_hole = np.any(gap_slice_hole > 0)
                has_retina = np.any(gap_slice_retina > 0)
                mean_intensity = float(np.mean(gap_slice_raw)) if gap_slice_raw.size > 0 else 0.0
                
                # Position relative to choroid center:
                gap_mid = (gap_start + gap_end) / 2.0
                
                gap_records.append({
                    'gap_size': gap_size,
                    'has_hole': has_hole,
                    'has_retina': has_retina,
                    'mean_intensity': mean_intensity,
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'dist_to_anchor': gap_mid - anchor_mid
                })
                
        # Proposed merged-band rule:
        # Merge runs if the gap between them is small (< 15 px) and not separated by hole/retina.
        # Specifically, discard runs that are far above the main choroid band (> 25 px gap or hole).
        # We start with the lowest run (CSI is always the lower boundary of choroid) or the main run,
        # and merge adjacent runs upwards if the gap is < 15 px.
        # Alternatively: merge all runs whose gap is <= 15 px.
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
        
        # Pick the merged group closest to anchor_mid or containing the most pixels
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
        
    print(f"Auditing {len(test_lines)} test scans...")
    all_col_records = []
    all_gap_records = []
    
    for idx, p in enumerate(test_lines):
        res = analyze_scan(p)
        if res is not None:
            cols, gaps = res
            all_col_records.extend(cols)
            all_gap_records.extend(gaps)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(test_lines)} scans...")
            
    df_cols = pd.DataFrame(all_col_records)
    df_gaps = pd.DataFrame(all_gap_records)
    
    um_per_px = 3.75 # Nominal assumption
    
    print("\n" + "="*70)
    print("CHOROID THICKNESS DISTRIBUTION ACROSS TEST SET COLUMNS")
    print("="*70)
    print(f"Total choroid columns analyzed: {len(df_cols)}")
    print(f"Columns with 1 run:  {(df_cols['n_runs'] == 1).sum()} ({(df_cols['n_runs'] == 1).mean()*100:.2f}%)")
    print(f"Columns with >=2 runs: {(df_cols['n_runs'] >= 2).sum()} ({(df_cols['n_runs'] >= 2).mean()*100:.2f}%)")
    print(f"Columns with >=3 runs: {(df_cols['n_runs'] >= 3).sum()} ({(df_cols['n_runs'] >= 3).mean()*100:.2f}%)")
    
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
    print("ANALYSIS OF GAPS IN MULTI-RUN COLUMNS")
    print("="*70)
    print(f"Total gaps identified: {len(df_gaps)}")
    if len(df_gaps) > 0:
        print(f"Gap size Median: {df_gaps['gap_size'].median():.1f} px, IQR: [{df_gaps['gap_size'].quantile(0.25):.1f}, {df_gaps['gap_size'].quantile(0.75):.1f}]")
        print(f"Gap size Mean: {df_gaps['gap_size'].mean():.2f} px, Min: {df_gaps['gap_size'].min()}, Max: {df_gaps['gap_size'].max()}")
        print(f"Gaps <= 5 px:   {(df_gaps['gap_size'] <= 5).sum()} ({(df_gaps['gap_size'] <= 5).mean()*100:.2f}%)")
        print(f"Gaps <= 10 px:  {(df_gaps['gap_size'] <= 10).sum()} ({(df_gaps['gap_size'] <= 10).mean()*100:.2f}%)")
        print(f"Gaps <= 15 px:  {(df_gaps['gap_size'] <= 15).sum()} ({(df_gaps['gap_size'] <= 15).mean()*100:.2f}%)")
        print(f"Gaps > 15 px:   {(df_gaps['gap_size'] > 15).sum()} ({(df_gaps['gap_size'] > 15).mean()*100:.2f}%)")
        print(f"Gaps with hole pixels: {df_gaps['has_hole'].sum()} ({df_gaps['has_hole'].mean()*100:.2f}%)")
        print(f"Gaps with retina pixels: {df_gaps['has_retina'].sum()} ({df_gaps['has_retina'].mean()*100:.2f}%)")
        
        # Analyze gaps > 15 px vs <= 15 px
        small_gaps = df_gaps[df_gaps['gap_size'] <= 15]
        large_gaps = df_gaps[df_gaps['gap_size'] > 15]
        print(f"\nSmall gaps (<= 15 px) mean intensity: {small_gaps['mean_intensity'].mean():.1f} (dark lumens)")
        if len(large_gaps) > 0:
            print(f"Large gaps (> 15 px) count: {len(large_gaps)}, mean intensity: {large_gaps['mean_intensity'].mean():.1f}, has hole: {large_gaps['has_hole'].sum()}")

if __name__ == '__main__':
    main()

