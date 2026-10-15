"""
Training utilities: losses, metrics, training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


# ─── Losses ───────────────────────────────────────────────────────────────────

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Differentiable Dice loss.

    Args:
        pred_logits: (B, 1, H, W) raw logits
        target: (B, 1, H, W) binary labels in {0, 1}
        smooth: smoothing constant to avoid division by zero
    Returns:
        scalar loss in [0, 1] (0 = perfect)
    """
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def combined_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                  dice_weight: float = 1.0, bce_weight: float = 0.5) -> torch.Tensor:
    """Combined Dice + BCE loss."""
    d_loss = dice_loss(pred_logits, target)
    b_loss = F.binary_cross_entropy_with_logits(pred_logits, target)
    return dice_weight * d_loss + bce_weight * b_loss


# ─── Metrics ──────────────────────────────────────────────────────────────────

def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice coefficient (higher is better).

    Args:
        pred_logits: (B, 1, H, W)
        target: (B, 1, H, W)
        threshold: binarization threshold for predictions
    Returns:
        mean Dice score across the batch (float)
    """
    with torch.no_grad():
        pred = (torch.sigmoid(pred_logits) > threshold).float()
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return dice.mean().item()


# ─── Training step ────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device: torch.device,
    n_steps_range: Tuple[int, int] = (32, 64),
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch with random number of NCA steps.

    Using a random number of steps per batch encourages the model to produce
    stable outputs regardless of the exact step count.
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Randomize number of NCA steps (helps learn stable dynamics)
        n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)

        optimizer.zero_grad()
        logits = model(images, n_steps=n_steps)
        loss = combined_loss(logits, labels)
        loss.backward()

        # Gradient clipping (NCA can have long computation graphs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(logits, labels)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device: torch.device,
    n_steps: int = 64,
) -> Dict[str, float]:
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images, n_steps=n_steps)
        loss = combined_loss(logits, labels)

        total_loss += loss.item()
        total_dice += dice_score(logits, labels)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
    }
