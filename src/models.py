"""
Model architecture definitions and builder factory for choroid segmentation.

Supports:
- U-Net from scratch (no pretrained weights)
- U-Net with pretrained encoders (e.g., ResNet34)
- U-Net++ with attention mechanisms (e.g., scSE decoder attention)
- Attention U-Net
"""

from typing import Optional
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(
    model_type: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 1,
    classes: int = 1,
    decoder_attention_type: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to construct segmentation architectures.

    Args:
        model_type: 'unet', 'unetplusplus', or 'manet'
        encoder_name: e.g. 'resnet34', 'resnet50', 'efficientnet-b2'
        encoder_weights: 'imagenet' or None (for scratch training)
        in_channels: Number of input channels (1 for grayscale OCT)
        classes: Number of output segmentation masks (1 for choroid binary mask)
        decoder_attention_type: e.g. 'scse' or None
    """
    m_type = model_type.lower().replace("-", "").replace("_", "")

    if m_type == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=decoder_attention_type,
        )
    elif m_type in ("unetplusplus", "unetpp"):
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=decoder_attention_type,
        )
    elif m_type == "manet":
        return smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose from 'unet', 'unetplusplus', or 'manet'."
        )

