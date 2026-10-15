"""
Visualization utilities for NCA segmentation and self-repair experiments.

Reusable functions for plotting segmentation results and healing dynamics.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert logits to binary mask numpy array.

    Args:
        logits: (B, 1, H, W) or (1, H, W) or (H, W)
        threshold: binarization threshold

    Returns:
        mask: numpy array, same spatial shape, binary {0, 1}
    """
    with torch.no_grad():
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float().cpu().numpy()
    # Squeeze to 2D if single image
    return mask.squeeze()


def logits_to_prob(logits: torch.Tensor) -> np.ndarray:
    """Convert logits to probability map.

    Args:
        logits: tensor of any shape

    Returns:
        prob: numpy array of same shape, values in [0, 1]
    """
    with torch.no_grad():
        return torch.sigmoid(logits).cpu().numpy().squeeze()


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """Overlay a binary mask on a grayscale image.

    Args:
        image: (H, W) grayscale in [0, 1]
        mask: (H, W) binary {0, 1}
        alpha: transparency of the overlay
        color: RGB color for the mask

    Returns:
        overlay: (H, W, 3) RGB image
    """
    # Convert grayscale to RGB
    rgb = np.stack([image] * 3, axis=-1)  # (H, W, 3)

    # Create colored mask
    color_mask = np.zeros_like(rgb)
    for c in range(3):
        color_mask[:, :, c] = mask * color[c]

    # Blend
    overlay = rgb * (1 - alpha * mask[:, :, np.newaxis]) + \
              color_mask * alpha
    return np.clip(overlay, 0, 1)
