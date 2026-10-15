"""
Damage functions for corrupting the NCA cell-state grid.

Used to test the self-repair hypothesis: if we damage the hidden state
partway through inference, can the NCA heal itself in subsequent steps?
"""

import torch
import numpy as np
from typing import Tuple, Optional


def rectangular_damage(
    hidden: torch.Tensor,
    severity: float = 0.3,
    center: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Zero out a rectangular region of the hidden state.

    Args:
        hidden: (B, C, H, W) cell state tensor.
        severity: fraction of total image area to damage (0.0 to 1.0).
        center: (cy, cx) as fractions of H, W. If None, random center.
        seed: random seed for reproducibility.

    Returns:
        damaged: (B, C, H, W) with the rectangle set to zero.
    """
    B, C, H, W = hidden.shape
    area = H * W
    damage_area = int(area * severity)

    # Compute side length of a square with this area
    side = int(np.sqrt(damage_area))
    side = max(side, 1)

    rng = np.random.RandomState(seed)

    if center is None:
        # Random center, but ensure the rectangle fits
        cy = rng.randint(side // 2, H - side // 2) if side < H else H // 2
        cx = rng.randint(side // 2, W - side // 2) if side < W else W // 2
    else:
        cy = int(center[0] * H)
        cx = int(center[1] * W)

    y0 = max(0, cy - side // 2)
    y1 = min(H, y0 + side)
    x0 = max(0, cx - side // 2)
    x1 = min(W, x0 + side)

    damaged = hidden.clone()
    damaged[:, :, y0:y1, x0:x1] = 0.0

    return damaged


def create_damage_fn(severity: float, center=None, seed: int = 42):
    """Create a damage function closure for use with model.forward_with_damage().

    Args:
        severity: fraction of image area to damage
        center: optional (cy, cx) as fractions
        seed: random seed

    Returns:
        callable that takes (B, C, H, W) -> (B, C, H, W)
    """
    def fn(hidden):
        return rectangular_damage(hidden, severity=severity, center=center, seed=seed)
    return fn


def get_damage_mask(
    H: int, W: int,
    severity: float = 0.3,
    center: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Get a binary mask showing the damaged region (for visualization).

    Returns:
        mask: (H, W) numpy array, 1 = damaged, 0 = intact
    """
    area = H * W
    damage_area = int(area * severity)
    side = max(int(np.sqrt(damage_area)), 1)

    rng = np.random.RandomState(seed)

    if center is None:
        cy = rng.randint(side // 2, H - side // 2) if side < H else H // 2
        cx = rng.randint(side // 2, W - side // 2) if side < W else W // 2
    else:
        cy = int(center[0] * H)
        cx = int(center[1] * W)

    y0 = max(0, cy - side // 2)
    y1 = min(H, y0 + side)
    x0 = max(0, cx - side // 2)
    x1 = min(W, x0 + side)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


if __name__ == "__main__":
    # Quick test
    h = torch.randn(2, 16, 64, 64)
    for sev in [0.1, 0.3, 0.5]:
        d = rectangular_damage(h, severity=sev, seed=42)
        n_zero = (d == 0).all(dim=1).float().mean().item()
        print("Severity {:.0%}: fraction of cells zeroed = {:.1%}".format(sev, n_zero))
