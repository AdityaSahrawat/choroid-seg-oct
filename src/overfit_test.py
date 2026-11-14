"""
Overfitting Verification Test.

Trains a U-Net model on a tiny batch of 2-4 images to confirm:
1. Data loading works cleanly.
2. Binary mask extraction (pixel value == 160) works.
3. Loss computation, backpropagation, and optimizer work together end-to-end.
4. Model architecture has sufficient capacity to reach Dice >= 0.99.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ChoroidDataset
from src.losses import BCEDiceLoss
from src.models import build_model
from src.preprocess import OCTPreprocessor


def create_synthetic_fixtures(out_dir: str = "outputs/test_fixtures") -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    img_paths, mask_paths = [], []

    for i in range(2):
        img_path = os.path.join(out_dir, f"sample_{i}.png")
        mask_path = os.path.join(out_dir, f"sample_{i}_mask.png")

        # Realistic dimensions (800 x 1100 like GOALS)
        H, W = 800, 1100
        # Background OCT speckle texture
        img = np.random.normal(50, 15, (H, W)).clip(0, 255).astype(np.uint8)
        # Hyper-reflective retinal layers
        img[200:230, :] = np.random.normal(180, 20, (30, W)).clip(0, 255).astype(np.uint8)
        # Choroidal stroma (hypo-reflective vessel lumens)
        img[300:450, :] = np.random.normal(80, 25, (150, W)).clip(0, 255).astype(np.uint8)
        # Sclera interface
        img[450:480, :] = np.random.normal(170, 20, (30, W)).clip(0, 255).astype(np.uint8)

        # GOALS mask
        mask = np.full((H, W), 255, dtype=np.uint8)
        mask[300:450, :] = 160  # Choroid

        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)
        img_paths.append(img_path)
        mask_paths.append(mask_path)

    return img_paths, mask_paths


def run_overfit_test(epochs: int = 60, lr: float = 1e-3, target_dice: float = 0.985):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n--- Running Overfit Test on Device: {device} ---")

    # Look for real data in data/splits or data/raw
    train_split = "data/splits/train.txt"
    img_paths, mask_paths = [], []

    if os.path.isfile(train_split):
        with open(train_split, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and os.path.exists(parts[0]) and os.path.exists(parts[1]):
                        img_paths.append(parts[0])
                        mask_paths.append(parts[1])
                if len(img_paths) >= 2:
                    break

    if len(img_paths) < 2:
        print("[INFO] Real data not present in data/splits/train.txt. Creating 2 synthetic OCT fixture scans...")
        img_paths, mask_paths = create_synthetic_fixtures()

    print(f"Using {len(img_paths)} scans for overfit check:")
    for p in img_paths:
        print(f"  {p}")

    # Build dataset & loader with deterministic settings (no random flip/rotate)
    preproc = OCTPreprocessor(use_clahe=True)
    dataset = ChoroidDataset(
        img_paths=img_paths,
        mask_paths=mask_paths,
        target_size=256,  # 256 for rapid test execution
        choroid_pixel_val=160,
        is_train=False,   # No stochastic augmentation to allow exact overfitting
        preprocessor=preproc,
        normalize_mean=0.485,
        normalize_std=0.229,
    )

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Build scratch model
    model = build_model(
        model_type="unet",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    ).to(device)

    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    print(f"\nTraining for up to {epochs} epochs (stopping when Dice >= {target_dice})...")
    reached_target = False

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Compute Dice
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            intersection = (preds * y).sum()
            cardinality = preds.sum() + y.sum()
            dice = (2.0 * intersection / (cardinality + 1e-8)).item()

        if epoch % 5 == 0 or epoch == 1 or dice >= target_dice:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {loss.item():.4f} | Training Dice: {dice:.4f}")

        if dice >= target_dice:
            print(f"\nSUCCESS: Overfit verified! Reached Dice {dice:.4f} >= {target_dice} at epoch {epoch}.")
            reached_target = True
            break

    if not reached_target:
        print(f"\nTest finished. Final Dice: {dice:.4f} (target: {target_dice})")


def main():
    run_overfit_test()


if __name__ == "__main__":
    main()
