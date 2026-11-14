"""
Loss functions for Choroid Segmentation.

Includes:
- Dice Loss (soft Dice for binary segmentation)
- Binary Cross Entropy with spatial boundary weights
- Combined Total Loss (BCE + Dice + Boundary-weighted BCE)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Computes loss over spatial dimensions (height, width).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape (B, 1, H, W)
            targets: Binary ground truth mask of shape (B, 1, H, W)
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return (1.0 - dice_score).mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss with optional spatial boundary weighting.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(eps=eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output
            targets: (B, 1, H, W) ground truth in {0, 1}
            weights: Optional (B, 1, H, W) spatial boundary weights
        """
        if weights is not None:
            # Spatially weighted binary cross entropy
            pixel_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            bce = (pixel_bce * weights).mean()
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        dice = self.dice_loss(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice

