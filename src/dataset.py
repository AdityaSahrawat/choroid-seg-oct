"""
PyTorch Dataset for Choroid OCT Segmentation.

Handles:
- Loading raw OCT B-scans and ground truth multi-class masks.
- Converting target label (e.g., pixel value 160 for GOALS) to binary choroid mask.
- Preprocessing (CLAHE, optional denoising).
- Augmentations (rotation <= 10 deg, horizontal flip, brightness/contrast, noise; no vertical flip).
- Boundary distance weight map computation directly on the augmented, resized mask.
- Standard ImageNet or custom mean/std normalization.
- Native image resolution tracking (orig_h, orig_w) for uncompressed thickness measurement.
"""

from typing import List, Optional, Tuple, Union
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from scipy.ndimage import distance_transform_edt

from src.preprocess import OCTPreprocessor


def compute_boundary_weight_map(
    mask: np.ndarray,
    w0: float = 5.0,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Computes spatial weight map giving exponentially higher weight to pixels near mask boundaries.
    Must be computed directly on the final augmented/resized mask matching network input.

    Formula:
        d = min(EDT(M), EDT(1 - M))
        W = 1.0 + w0 * exp(-(d^2) / (2 * sigma^2))
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0 or m.sum() == m.size:
        return np.ones(mask.shape, dtype=np.float32)

    d_in = distance_transform_edt(m)
    d_out = distance_transform_edt(1 - m)
    d = np.maximum(d_in + d_out - 1.0, 0.0)
    weight = 1.0 + w0 * np.exp(-(d ** 2) / (2.0 * (sigma ** 2)))
    return weight.astype(np.float32)


class ChoroidDataset(Dataset):
    """
    Dataset class for Choroid segmentation from retinal OCT scans.
    """

    def __init__(
        self,
        img_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        target_size: Union[int, Tuple[int, int], List[int]] = 512,
        choroid_pixel_val: int = 160,
        mask_tolerance: int = 0,
        is_train: bool = True,
        preprocessor: Optional[OCTPreprocessor] = None,
        compute_boundary_weights: bool = False,
        boundary_w0: float = 5.0,
        boundary_sigma: float = 5.0,
        normalize_mean: Optional[float] = 0.485,
        normalize_std: Optional[float] = 0.229,
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.choroid_pixel_val = choroid_pixel_val
        self.mask_tolerance = mask_tolerance
        self.is_train = is_train
        self.compute_boundary_weights = compute_boundary_weights
        self.boundary_w0 = boundary_w0
        self.boundary_sigma = boundary_sigma
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = (int(target_size[0]), int(target_size[1]))

        self.preprocessor = preprocessor or OCTPreprocessor()

        # Build Albumentations transform pipeline
        # Note: Vertical flips are strictly forbidden because retinal layers follow a strict anatomical order.
        # Horizontal flip is valid because image and mask flip synchronously, preserving layer depths.
        aug_list = [A.Resize(height=self.target_size[0], width=self.target_size[1], interpolation=cv2.INTER_LINEAR)]
        if self.is_train:
            aug_list += [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.GaussNoise(p=0.2),
            ]
        self.transform = A.Compose(aug_list)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx] if (self.mask_paths is not None and len(self.mask_paths) > idx) else None

        # Check for OIMHS composite side-by-side format (left: raw OCT, right: color-coded mask)
        is_composite_request = (mask_path is None or mask_path == img_path or mask_path == "composite")

        if is_composite_request:
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise FileNotFoundError(f"Failed to load image at: {img_path}")

            H, W_full = bgr.shape[:2]
            W_mid = W_full // 2
            # Left half is raw OCT B-scan
            img_raw = cv2.cvtColor(bgr[:, :W_mid], cv2.COLOR_BGR2GRAY)
            orig_h, orig_w = img_raw.shape[:2]
            img = self.preprocessor(img_raw)

            # Right half contains ground truth: Choroid is Yellow (BGR [0, 255, 255], RGB [255, 255, 0])
            # In OIMHS:
            # - Yellow [0, 255, 255] is Choroid (lies beneath Bruch's membrane)
            # - Green [0, 255, 0] is Neurosensory Retina
            # - Red [0, 0, 255] is Macular Hole
            # - Blue [255, 0, 0] is Intraretinal Cysts
            right_half = bgr[:, W_mid:]
            binary_mask = (
                (right_half[:, :, 0] == 0) &
                (right_half[:, :, 1] == 255) &
                (right_half[:, :, 2] == 255)
            ).astype(np.uint8)
            has_mask = True
        else:
            img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                raise FileNotFoundError(f"Failed to load image at: {img_path}")
            orig_h, orig_w = img_raw.shape[:2]
            img = self.preprocessor(img_raw)

            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                raise FileNotFoundError(f"Failed to load mask at: {mask_path}")

            # Extract binary choroid mask: GOALS pixel value 160 -> 1, otherwise 0
            if self.mask_tolerance > 0:
                binary_mask = (
                    (raw_mask >= self.choroid_pixel_val - self.mask_tolerance) &
                    (raw_mask <= self.choroid_pixel_val + self.mask_tolerance)
                ).astype(np.uint8)
            else:
                binary_mask = (raw_mask == self.choroid_pixel_val).astype(np.uint8)
            has_mask = True

        if has_mask:
            # 3. Apply augmentations & resizing synchronously to image and mask
            transformed = self.transform(image=img, mask=binary_mask)
            img_t = transformed["image"]
            mask_t = transformed["mask"]

            # 4. Compute boundary weight map AFTER resize/augmentation on the exact mask seen by the loss
            if self.compute_boundary_weights:
                weight_t = compute_boundary_weight_map(
                    mask_t,
                    w0=self.boundary_w0,
                    sigma=self.boundary_sigma,
                )
            else:
                weight_t = None
        else:
            transformed = self.transform(image=img)
            img_t = transformed["image"]
            mask_t = None
            weight_t = None

        # 5. Convert image to float tensor with normalization
        x_norm = img_t.astype(np.float32) / 255.0
        if self.normalize_mean is not None and self.normalize_std is not None:
            x_norm = (x_norm - self.normalize_mean) / self.normalize_std

        x = torch.from_numpy(x_norm).float().unsqueeze(0)

        item = {
            "image": x,
            "orig_hw": (orig_h, orig_w),
            "img_path": img_path,
        }

        if mask_t is not None:
            y = torch.from_numpy(mask_t).float().unsqueeze(0)
            item["mask"] = y
            item["orig_mask"] = binary_mask

        if weight_t is not None:
            w = torch.from_numpy(weight_t).float().unsqueeze(0)
            item["weight"] = w

        return item
