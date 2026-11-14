"""
Preprocessing utilities for retinal OCT B-scans.

Includes:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Speckle noise reduction (median or bilateral filtering)
- Normalization and standardization
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class OCTPreprocessor:
    """Preprocessor for retinal OCT B-scan images."""

    def __init__(
        self,
        use_clahe: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        denoise_method: Optional[str] = None,  # None, "median", "bilateral"
        median_kernel: int = 3,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
    ):
        self.use_clahe = use_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.denoise_method = denoise_method
        self.median_kernel = median_kernel
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self._clahe = None

    @property
    def clahe(self):
        if self._clahe is None and self.use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
        return self._clahe

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_clahe"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._clahe = None

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply light speckle reduction if enabled."""
        if self.denoise_method == "median":
            return cv2.medianBlur(img, self.median_kernel)
        elif self.denoise_method == "bilateral":
            return cv2.bilateralFilter(
                img,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )
        return img

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Enhance local contrast to sharpen the choroid-sclera interface."""
        if self.clahe is not None:
            return self.clahe.apply(img)
        return img

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline on single 8-bit grayscale image."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Speckle reduction (optional)
        img = self.denoise(img)

        # 2. Contrast enhancement
        img = self.apply_clahe(img)

        return img

