"""
Main training script for Choroid OCT Segmentation.

Features:
- Driven by YAML configuration files.
- Supports Apple Silicon MPS acceleration.
- Automatic dataset loading and validation.
- Full validation at original resolution:
  - Ground truth boundaries extracted strictly WITHOUT smoothing/interpolation.
  - Predictions extracted with Savitzky-Golay smoothing.
  - Evaluates Dice, IoU, Upper Boundary MAE, Lower Boundary (CSI) MAE, and Thickness MAE.
- TensorBoard logging and early stopping.
- Checkpointing best model based on validation Dice score.
"""

import argparse
import json
import os
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.dataset import ChoroidDataset
from src.evaluate import (
    compute_boundary_errors,
    compute_dice_iou,
    compute_thickness_metrics,
)
from src.losses import BCEDiceLoss
from src.models import build_model
from src.postprocess import (
    clean_mask,
    extract_gt_boundaries,
    extract_pred_boundaries,
)
from src.preprocess import OCTPreprocessor
from src.thickness import compute_thickness_profile


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_setting: str = "auto") -> torch.device:
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_setting)


def load_split_file(filepath: str) -> Tuple[List[str], List[str]]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Split file '{filepath}' does not exist.")
    img_paths, mask_paths = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            img_paths.append(parts[0].strip())
            mask_paths.append(parts[1].strip() if len(parts) > 1 else None)
    return img_paths, mask_paths


def cv2_resize_mask(mask: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
    orig_h, orig_w = orig_hw
    return cv2.resize((mask > 0.5).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    um_per_px: float = None,
    smooth_window: int = 31,
    poly_order: int = 2,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    val_loss_sum = 0.0
    dice_list, iou_list = [], []
    upper_mae_list, lower_mae_list = [], []
    thickness_mae_list = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            w = batch.get("weight", None)
            if w is not None:
                w = w.to(device)

            logits = model(x)
            loss = criterion(logits, y, w)
            val_loss_sum += loss.item() * x.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = y.cpu().numpy()
            orig_hws = batch["orig_hw"]

            for i in range(x.size(0)):
                prob_i = probs[i, 0]
                target_i = targets[i, 0]
                orig_hw = (int(orig_hws[0][i]), int(orig_hws[1][i]))

                # Clean prediction mask and reconstruct at native resolution
                pred_mask = clean_mask(prob_i, orig_hw=orig_hw, threshold=threshold)

                # Ground truth mask resized to native resolution (nearest-neighbor)
                gt_mask = cv2_resize_mask(target_i, orig_hw)

                # 1. Segmentation metrics
                d, iou = compute_dice_iou(pred_mask, gt_mask)
                dice_list.append(d)
                iou_list.append(iou)

                # 2. Extract boundaries:
                # Ground truth strictly raw without smoothing or interpolation
                top_gt, bot_gt, valid_gt = extract_gt_boundaries(gt_mask)
                # Prediction with Savitzky-Golay smoothing
                top_pred, bot_pred, valid_pred = extract_pred_boundaries(
                    pred_mask, smooth_window=smooth_window, poly_order=poly_order
                )

                # Boundary errors
                b_errs = compute_boundary_errors(
                    top_pred, bot_pred, top_gt, bot_gt, um_per_px=um_per_px
                )
                if not np.isnan(b_errs["upper_mae_px"]):
                    upper_mae_list.append(b_errs["upper_mae_px"])
                if not np.isnan(b_errs["lower_mae_px"]):
                    lower_mae_list.append(b_errs["lower_mae_px"])

                # 3. Thickness metrics
                t_pred, _ = compute_thickness_profile(top_pred, bot_pred, valid_pred, um_per_px)
                t_gt, _ = compute_thickness_profile(top_gt, bot_gt, valid_gt, um_per_px)

                t_metrics = compute_thickness_metrics(t_pred, t_gt)
                if not np.isnan(t_metrics["thickness_mae"]):
                    thickness_mae_list.append(t_metrics["thickness_mae"])

    n_samples = len(val_loader.dataset)
    return {
        "val_loss": val_loss_sum / max(n_samples, 1),
        "val_dice": float(np.mean(dice_list)) if dice_list else 0.0,
        "val_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "val_upper_mae_px": float(np.mean(upper_mae_list)) if upper_mae_list else 0.0,
        "val_lower_mae_px": float(np.mean(lower_mae_list)) if lower_mae_list else 0.0,
        "val_thickness_mae": float(np.mean(thickness_mae_list)) if thickness_mae_list else 0.0,
    }


def train(cfg: dict) -> None:
    exp_name = cfg.get("name", "experiment")
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = get_device(cfg.get("device", "auto"))
    print(f"[{exp_name}] Using device: {device} (reproducible within run-to-run variance on MPS)")

    # Output directory setup
    output_dir = os.path.join(cfg.get("output_dir", "outputs"), exp_name)
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Preprocessor
    prep_cfg = cfg.get("preprocess", {})
    preprocessor = OCTPreprocessor(
        use_clahe=prep_cfg.get("use_clahe", True),
        clip_limit=prep_cfg.get("clip_limit", 2.0),
        tile_grid_size=tuple(prep_cfg.get("tile_grid_size", [8, 8])),
        denoise_method=prep_cfg.get("denoise_method", None),
    )

    # Data Splits
    data_cfg = cfg.get("data", {})
    train_imgs, train_masks = load_split_file(data_cfg["train_split"])
    val_imgs, val_masks = load_split_file(data_cfg["val_split"])
    print(f"Loaded {len(train_imgs)} training scans and {len(val_imgs)} validation scans.")

    compute_boundary_w = data_cfg.get("compute_boundary_weights", False)
    target_size = data_cfg.get("target_size", 512)
    norm_mean = data_cfg.get("normalize_mean", 0.485)
    norm_std = data_cfg.get("normalize_std", 0.229)

    train_ds = ChoroidDataset(
        img_paths=train_imgs,
        mask_paths=train_masks,
        target_size=target_size,
        choroid_pixel_val=data_cfg.get("choroid_pixel_val", 160),
        mask_tolerance=data_cfg.get("mask_tolerance", 0),
        is_train=True,
        preprocessor=preprocessor,
        compute_boundary_weights=compute_boundary_w,
        boundary_w0=data_cfg.get("boundary_w0", 5.0),
        boundary_sigma=data_cfg.get("boundary_sigma", 5.0),
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )

    val_ds = ChoroidDataset(
        img_paths=val_imgs,
        mask_paths=val_masks,
        target_size=target_size,
        choroid_pixel_val=data_cfg.get("choroid_pixel_val", 160),
        mask_tolerance=data_cfg.get("mask_tolerance", 0),
        is_train=False,
        preprocessor=preprocessor,
        compute_boundary_weights=compute_boundary_w,
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model_cfg = cfg.get("model", {})
    model = build_model(
        model_type=model_cfg.get("model_type", "unet"),
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 1),
        classes=model_cfg.get("classes", 1),
        decoder_attention_type=model_cfg.get("decoder_attention_type", None),
    ).to(device)

    # Loss, Optimizer, Scheduler
    t_cfg = cfg["training"]
    criterion = BCEDiceLoss(
        bce_weight=t_cfg.get("bce_weight", 1.0),
        dice_weight=t_cfg.get("dice_weight", 1.0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg.get("lr", 1e-4)),
        weight_decay=float(t_cfg.get("weight_decay", 1e-4)),
    )

    epochs = t_cfg.get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(t_cfg.get("min_lr", 1e-6))
    )

    thickness_cfg = cfg.get("thickness", {})
    um_per_px = thickness_cfg.get("um_per_px", None)

    best_val_dice = 0.0
    patience = t_cfg.get("early_stopping_patience", 20)
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", mininterval=5.0)
        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            w = batch.get("weight", None)
            if w is not None:
                w = w.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y, w)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_train_loss = train_loss_sum / len(train_loader.dataset)

        # Validation
        val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            um_per_px=um_per_px,
            smooth_window=thickness_cfg.get("smooth_window", 31),
            poly_order=thickness_cfg.get("poly_order", 2),
            threshold=cfg.get("threshold", 0.5),
        )

        val_dice = val_metrics["val_dice"]
        val_loss = val_metrics["val_loss"]

        # TensorBoard
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/val_dice", val_dice, epoch)
        writer.add_scalar("Metrics/val_iou", val_metrics["val_iou"], epoch)
        writer.add_scalar("Metrics/val_upper_mae_px", val_metrics["val_upper_mae_px"], epoch)
        writer.add_scalar("Metrics/val_lower_mae_px", val_metrics["val_lower_mae_px"], epoch)
        writer.add_scalar("Metrics/val_thickness_mae", val_metrics["val_thickness_mae"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | Val IoU: {val_metrics['val_iou']:.4f} | "
            f"Bruch's Membrane MAE: {val_metrics['val_upper_mae_px']:.2f} px | CSI MAE: {val_metrics['val_lower_mae_px']:.2f} px | "
            f"Thickness MAE: {val_metrics['val_thickness_mae']:.2f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                best_path,
            )
            print(f"  --> Saved new best checkpoint (Val Dice: {val_dice:.4f}) to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    writer.close()
    print(f"\nTraining completed. Best Validation Dice: {best_val_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Choroid OCT Segmentation network.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
