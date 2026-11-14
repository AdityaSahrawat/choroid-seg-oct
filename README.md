# Choroid Segmentation and Thickness Measurement (retinal OCT)

Automated deep learning pipeline for segmentation of the choroid and measurement of choroidal thickness from retinal Optical Coherence Tomography (OCT) B-scans (specifically circumpapillary scans from the MICCAI 2022 GOALS challenge).

Project dependency and environment management is managed entirely with **`uv`** via [`pyproject.toml`](file:///Users/adityasahrawat/dev/projects/choroid-seg-oct/pyproject.toml) and [`uv.lock`](file:///Users/adityasahrawat/dev/projects/choroid-seg-oct/uv.lock).

---

## Key Pipeline Features

1. **Preprocessing & Speckle Handling**:
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the faint lower choroid boundary (Choroid-Sclera Interface, CSI).
   - Optional light speckle filtering (median / bilateral).
2. **Data Integrity & Splitting**:
   - Strictly splits by **eye / subject**, never by individual B-scans, preventing clinical data leakage.
   - Extracts choroid masks using the GOALS challenge label (pixel value `160`).
   - Supports both standard train/val/test partitions and **5-Fold Cross-Validation by Eye** (`--cv_folds 5`).
3. **Model Architectures**:
   - **Baseline 1:** U-Net trained from scratch (`configs/unet_scratch.yaml`).
   - **Baseline 2:** U-Net with ImageNet-pretrained ResNet-34 encoder (`configs/unet_r34.yaml`).
   - **Improved Model:** U-Net++ with ResNet-34 encoder and scSE attention + boundary-weighted loss (`configs/unetpp_r34_scse.yaml`).
4. **Boundary Extraction & Smoothing**:
   - Probability map resized back to native scan resolution (`orig_h, orig_w`).
   - **Ground-Truth Boundaries:** Extracted strictly **without smoothing and without interpolation** (`extract_gt_boundaries`).
   - **Predicted Boundaries:** Extracted with largest connected component filtering, binary hole filling, and Savitzky-Golay smoothing (`extract_pred_boundaries`).
   - An on/off ablation flag (`--no_postprocess`) allows evaluating raw thresholded predictions directly against smoothed predictions on the same checkpoint.
5. **Choroidal Thickness & Sector Summaries**:
   - Computes vertical distance $t(c) = z_{\mathrm{lower}}(c) - z_{\mathrm{upper}}(c)$.
   - Configurable axial resolution `um_per_px`:
     - When `um_per_px: null` (default), reports thickness **strictly in pixels (`px`)** with zero assumptions.
     - When `um_per_px` is provided (e.g., $3.87$), converts to micrometers ($\mu m$).
   - Reports sector means strictly as `"Sector 1"`, `"Sector 2"`, `"Sector 3"`, `"Sector 4"` to avoid unverified anatomical quadrant assumptions.
6. **Multi-Level Clinical Evaluation**:
   - **Mask Overlap:** Dice Similarity Coefficient (DSC), Intersection over Union (IoU), 95th-percentile Hausdorff Distance (HD95).
   - **Boundary Error:** Mean Absolute Error (MAE) for upper boundary and lower boundary (CSI) separately.
   - **Thickness Agreement:**
     - Primary headline agreement: **One point per scan** Bland-Altman mean bias & 95% Limits of Agreement, and cohort Pearson $r$ on per-scan means.
     - Secondary diagnostic: Pooled-column Bland-Altman plot.
   - **Failure Analysis:** Automatically extracts and plots the top 3 worst scans ranked by lower boundary (CSI) error.

---

## Directory Structure

```text
choroid-seg-oct/
├── configs/
│   ├── unet_scratch.yaml            # Baseline U-Net from scratch
│   ├── unet_r34.yaml                # Pretrained ResNet-34 U-Net
│   ├── unetpp_r34_scse.yaml         # U-Net++ with scSE attention & boundary loss
│   └── ablations/
│       ├── unet_r34_boundary_loss.yaml   # Boundary loss ablation on U-Net
│       ├── unet_r34_no_clahe.yaml        # CLAHE disabled ablation
│       └── unet_r34_aspect_ratio.yaml    # Non-square 576x800 aspect ratio ablation
├── data/
│   ├── raw/                         # Place GOALS images and Layer_Masks here (not committed)
│   └── splits/                      # train.txt, val.txt, test.txt, cv_5folds/
├── src/
│   ├── __init__.py
│   ├── check_masks.py               # Inspects ground truth mask unique pixel values
│   ├── dataset.py                   # PyTorch Dataset, augmentation, boundary weights, normalization
│   ├── evaluate.py                  # Dice, IoU, HD95, boundary MAE, per-scan Bland-Altman
│   ├── losses.py                    # Soft Dice loss, BCE, boundary-weighted loss
│   ├── models.py                    # smp factory (U-Net, U-Net++, Attention)
│   ├── overfit_test.py              # Overfit sanity check on 2-4 scans
│   ├── postprocess.py               # Raw GT extraction, pred LCC, hole fill, SavGol
│   ├── preprocess.py                # CLAHE, speckle filtering
│   ├── run_experiments.py           # Multi-seed runner and results aggregator
│   ├── split_data.py                # Split by eye ID & 5-fold CV generator
│   ├── test_eval.py                 # Test evaluation, postprocess ablation, failure case figures
│   ├── thickness.py                 # Column-wise thickness and sector summaries
│   └── train.py                     # Main training loop with TensorBoard & early stopping
├── tests/
│   └── test_pipeline.py             # 10 unit tests covering all components
├── outputs/                         # Checkpoints, figures, metrics tables
├── pyproject.toml                   # UV project specification & dependencies
├── uv.lock                          # UV deterministic lockfile
└── README.md
```

---

## Step-by-Step Workflow

### 1. Environment Synchronization & Tests

```bash
# Sync packages
uv sync

# Run the 10 unit tests
uv run pytest tests/
```

### 2. Dataset Verification & Splitting

The pipeline natively supports both:
1. **OIMHS Dataset (Macular OCT)** located at `data/raw/OIMHS dataset`:
   - 3,859 composite scans ($512 \times 1024$) where left half is raw OCT and right half is 4-color manual ground truth (pure green `[0, 255, 0]` is Choroid).
   - Eye-to-Patient mapping via `Demographics of the participants.xlsx` prevents patient leakage across bilateral eyes.
2. **GOALS Challenge Dataset (Circumpapillary OCT)**:
   - Standalone B-scans with separate mask files where pixel value `160` is Choroid.

```bash
# Verify mask values across OIMHS scans
uv run python src/check_masks.py --mask_dir "data/raw/OIMHS dataset"

# Generate patient-level splits and 5-fold cross-validation for OIMHS
uv run python src/split_data.py --oimhs_dir "data/raw/OIMHS dataset" --cv_folds 5
```

### 3. Rapid Overfit Sanity Check

```bash
uv run python src/overfit_test.py
```

### 4. Training

```bash
# Pilot training (200 train / 50 val scans, 5 epochs for rapid verification)
uv run python src/train.py --config configs/oimhs_pilot.yaml

# Full dataset training (ResNet-34 U-Net)
uv run python src/train.py --config configs/unet_r34.yaml

# Improved model (U-Net++ with scSE attention & boundary loss)
uv run python src/train.py --config configs/unetpp_r34_scse.yaml
```

Monitor training telemetry with TensorBoard:
```bash
uv run tensorboard --logdir outputs/
```

### 5. Test Set Evaluation & Post-Processing Ablation

```bash
# Standard evaluation with post-processing (LCC + Hole Fill + SavGol smoothing)
uv run python src/test_eval.py --checkpoint outputs/oimhs_pilot/best_model.pt --split_file data/splits/pilot/test.txt --num_visualizations 5

# Post-processing ablation (raw probability thresholding resized to native)
uv run python src/test_eval.py --checkpoint outputs/oimhs_pilot/best_model.pt --split_file data/splits/pilot/test.txt --no_postprocess --output_dir outputs/oimhs_pilot/test_eval_no_postprocess
```

### 6. Multi-Seed Sweeps Across 3 Seeds

```bash
uv run python src/run_experiments.py --configs configs/unet_r34.yaml configs/unetpp_r34_scse.yaml --seeds 42 43 44
```

---

## Verified Real-Data Headline Benchmark (Full Test Set: 697 Scans across 18 Independent Patients)

Evaluated using the ResNet-34 U-Net model trained on 2,609 scans with LCC-only post-processing (hole filling and boundary smoothing disabled):

| Metric | Headline Performance (LCC-Only) | Raw Predictions (No Post-Process) | Unit |
| :--- | :--- | :--- | :--- |
| **Dice Similarity (DSC)** | **0.9920 ± 0.0054** | 0.9920 ± 0.0054 | — |
| **Intersection over Union (IoU)** | **0.9842 ± 0.0105** | 0.9842 ± 0.0105 | — |
| **Bruch's Membrane MAE** | **2.192 px** | 2.207 px | px |
| **Lower Boundary (CSI) MAE** | **1.880 px** | 1.897 px | px |
| **Choroidal Thickness MAE** | **2.095 px** | 2.098 px | px |
| **Mask Contour HD95 (2D)** | **1.69 px** | 1.69 px | px |
| **Bruch's Membrane HD95 (1D)** | **6.30 px** | 6.30 px | px |
| **CSI Lower Boundary HD95 (1D)** | **6.26 px** | 6.26 px | px |
| **Cohort Pearson $r$ (Per-Scan)** | **0.9795** | 0.9795 | — |
| **Bland-Altman Per-Scan Mean Bias** | **-0.040 px** (95% LoA: [-4.29, +4.22] px) | -0.040 px | px |

### Robustness Quality Breakdown (Full Test Set)
* **Good Quality Scans ($n = 567$):** Dice = **0.9923**, Bruch's Membrane MAE = **1.99 px**, CSI MAE = **1.81 px**, Thickness MAE = **1.96 px**
* **Poor Quality Scans ($n = 130$):** Dice = **0.9906**, Bruch's Membrane MAE = **3.08 px**, CSI MAE = **2.18 px**, Thickness MAE = **2.68 px**

### Subfoveal Choroidal Thickness (SFCT) Across 19 Foveal Eyes
* **Mean Subfoveal Ground Truth Thickness:** **9.95 px**
* **Mean Subfoveal Predicted Thickness:** **9.53 px**
* **Subfoveal Thickness MAE (Main Run):** **1.37 px**
* **Subfoveal MAE (Pixel Count Cross-Check):** **3.42 px**
* **Parafoveal Thickness MAE (Main Run):** **1.41 px**

---

## Step-by-Step Post-Processing Ablation (697 Scans across 18 Independent Patients)

Paired Wilcoxon signed-rank tests computed on per-patient mean Thickness MAE across $N=18$ independent patients:

| Configuration | Dice | Bruch's MAE | CSI MAE | Thickness MAE | Pooled Col SD | Paired Wilcoxon ($N=18$ Patients) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Raw (No post-processing)** | **0.9920** | 2.207 px | 1.897 px | 2.098 px | 8.08 px | *Baseline* |
| **2. LCC only (Final Pipeline)** | **0.9920** | **2.192 px** | **1.880 px** | **2.095 px** | **8.08 px** | $p = 0.3173$ *(not significant)* |
| **3. LCC + Hole Fill** | 0.9770 | 7.066 px | 4.592 px | 11.223 px | 24.58 px | $p < 0.001^*$ *(statistically worse)* |
| **4. LCC + Hole + SavGol ($w=7$)** | 0.9770 | 7.124 px | 4.629 px | 11.263 px | 24.56 px | $p < 0.001^*$ *(statistically worse)* |
| **5. LCC + Hole + SavGol ($w=11$)** | 0.9770 | 7.158 px | 4.648 px | 11.284 px | 24.55 px | $p < 0.001^*$ *(statistically worse)* |
| **6. LCC + Hole + SavGol ($w=15$)** | 0.9770 | 7.192 px | 4.665 px | 11.307 px | 24.55 px | $p < 0.001^*$ *(statistically worse)* |
| **7. LCC + Hole + SavGol ($w=31$)** | 0.9770 | 7.312 px | 4.723 px | 11.402 px | 24.55 px | $p < 0.001^*$ *(statistically worse)* |
| **Bland-Altman Per-Scan 95% LoA** | **[-1.44 px, +2.03 px]** | [-1.44 px, +1.97 px] | px |
| **Pooled-Columns Bland-Altman SD** | **3.59 px** *(smoother column profiles)* | 4.24 px *(higher column variance)* | px |
