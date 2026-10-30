"""
Augmentation pipeline combining Albumentations (2D spatial/intensity transforms)
and TorchIO (MRI-specific intensity transforms).

Why two libraries?
- Albumentations: best-in-class 2D spatial augmentations (rotate, flip, elastic
  deform, grid distort, coarse dropout, etc.) — all applied consistently to both
  image AND mask.
- TorchIO: purpose-built for MRI — simulates scanner artefacts such as random
  bias field (intensity inhomogeneity), ghosting, noise, blurring, and motion that
  are specific to medical imaging.  Applied to the image only (not the mask).

Usage:
    from src.augmentations import get_train_augmentation, apply_augmentation
    aug = get_train_augmentation()
    img_aug, lbl_aug = apply_augmentation(aug, img_np, lbl_np)

Input:  img_np  - np.float32 (H, W), values in [0, 1]
        lbl_np  - np.float32 (H, W), values in {0, 1}
Output: same shapes and dtype, augmented.
"""

import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchio as tio


# ─── Albumentations pipeline ─────────────────────────────────────────────────

def get_albumentations_pipeline(p_spatial: float = 0.8,
                                 p_intensity: float = 0.7) -> A.Compose:
    """
    Compose of spatial and intensity transforms.

    Spatial transforms are applied to BOTH image and mask (paired).
    Intensity transforms are applied only to the image channel.

    Args:
        p_spatial:   probability of applying the entire spatial block.
        p_intensity: probability of applying the entire intensity block.

    Returns:
        A.Compose object accepting {'image': H×W uint8/float, 'mask': H×W uint8}.
    """
    spatial = A.OneOf([
        # Small rotations – hippocampus can appear at slightly different angles
        A.Rotate(limit=15, border_mode=0, p=1.0),
        # Elastic deformation – mimics soft-tissue breathing motion
        A.ElasticTransform(alpha=60, sigma=6, p=1.0),
        # Perspective distortion – simulates scanner table tilt
        A.Perspective(scale=(0.03, 0.07), p=1.0),
        # Optical-flow-like local distortion
        A.GridDistortion(num_steps=4, distort_limit=0.2, p=1.0),
    ], p=p_spatial)

    flips = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
    ])

    intensity = A.OneOf([
        # Simulate slight MRI contrast differences
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        # CLAHE (contrast limited adaptive histogram equalisation)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
        # Brightness / contrast shift
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                    contrast_limit=0.15, p=1.0),
        # Gaussian noise (simulate MRI thermal noise)
        A.GaussNoise(std_range=(0.005, 0.02), p=1.0),
    ], p=p_intensity)

    # CoarseDropout: randomly blank-out small rectangular regions in the image
    # (simulates k-space acquisition dropout or corrupted image patches)
    dropout = A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(4, 8),
        hole_width_range=(4, 8),
        fill=0.0,
        p=0.3,
    )

    return A.Compose([flips, spatial, intensity, dropout],
                     additional_targets={"mask": "mask"})


# ─── TorchIO MRI-specific transforms ─────────────────────────────────────────

def get_torchio_pipeline() -> tio.Compose:
    """
    MRI-specific intensity augmentations via TorchIO.
    Applied to the IMAGE only (not the mask).

    Transforms:
        RandomBiasField  – low-frequency intensity inhomogeneity (e.g. B1 field)
        RandomNoise      – additive Gaussian noise in MRI signal
        RandomBlur       – scanner PSF / Gibbs ringing approximation
        RandomGhosting   – motion-induced ghosting artefact (p=0.15 only)
    """
    return tio.Compose([
        tio.RandomBiasField(coefficients=0.3, p=0.5),
        tio.RandomNoise(mean=0.0, std=(0.0, 0.015), p=0.5),
        tio.RandomBlur(std=(0.0, 0.8), p=0.3),
        tio.RandomGhosting(num_ghosts=(1, 3), intensity=(0.3, 0.7), p=0.15),
    ])


# ─── Combined apply function ─────────────────────────────────────────────────

_albu_pipeline = None
_tio_pipeline = None


def _lazy_init():
    """Initialise pipelines once and cache them."""
    global _albu_pipeline, _tio_pipeline
    if _albu_pipeline is None:
        _albu_pipeline = get_albumentations_pipeline()
    if _tio_pipeline is None:
        _tio_pipeline = get_torchio_pipeline()


def apply_augmentation(img: np.ndarray, lbl: np.ndarray) -> tuple:
    """
    Apply the full augmentation pipeline to a single (image, mask) pair.

    Args:
        img: np.float32 (H, W), pixel values in [0, 1]
        lbl: np.float32 (H, W), binary mask {0.0, 1.0}

    Returns:
        img_aug: np.float32 (H, W)
        lbl_aug: np.float32 (H, W)
    """
    _lazy_init()

    H, W = img.shape

    # ── Step 1: Albumentations (spatial + some intensity, image + mask paired) ──
    # Albumentations expects uint8 masks; we multiply by 255, then divide back.
    lbl_uint8 = (lbl * 255).astype(np.uint8)

    result = _albu_pipeline(image=img, mask=lbl_uint8)
    img_albu = result["image"].astype(np.float32)        # still (H, W)
    lbl_albu = (result["mask"] / 255.0).astype(np.float32)

    # ── Step 2: TorchIO (MRI artefacts, image only) ────────────────────────────
    # TorchIO Subject expects 4D tensors: (C, W, H, D) with D=1 for 2D data
    img_4d = torch.from_numpy(img_albu).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    subject = tio.Subject(image=tio.ScalarImage(tensor=img_4d))
    subject_aug = _tio_pipeline(subject)
    img_tio = subject_aug["image"].data.squeeze(-1).squeeze(0).numpy()  # (H, W)

    # Clip back to [0, 1] (bias field can push values outside)
    img_tio = np.clip(img_tio, 0.0, 1.0).astype(np.float32)

    return img_tio, lbl_albu


def visualise_augmentations(img: np.ndarray, lbl: np.ndarray,
                             n_samples: int = 6, save_path: str = None):
    """
    Show a grid of augmented versions of one (image, mask) pair.
    Useful for visual sanity-checking the pipeline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, n_samples + 1, figsize=(3 * (n_samples + 1), 6))

    # Column 0: original
    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original", fontsize=9)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.6)
    axes[1, 0].imshow(lbl, cmap="Reds", vmin=0, vmax=1, alpha=0.5)
    axes[1, 0].set_title("GT Mask", fontsize=9)
    axes[1, 0].axis("off")

    for col in range(1, n_samples + 1):
        img_aug, lbl_aug = apply_augmentation(img, lbl)
        axes[0, col].imshow(img_aug, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Aug {col}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(img_aug, cmap="gray", vmin=0, vmax=1, alpha=0.6)
        axes[1, col].imshow(lbl_aug, cmap="Reds", vmin=0, vmax=1, alpha=0.5)
        axes[1, col].axis("off")

    plt.suptitle("Augmentation preview (top=image, bottom=image+mask)", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved augmentation preview to {save_path}")
    else:
        plt.show()
    plt.close()

