"""
PyTorch Dataset for preprocessed 2D hippocampus slices.

Loads numpy arrays saved by preprocess_data.py and serves (image, label) pairs
as tensors with shapes:
    image: (1, H, W) float32 in [0, 1]
    label: (1, H, W) float32 in {0, 1}

Augmentation strategy (training split only):
    - Albumentations: spatial transforms (rotation, elastic, perspective, grid
      distortion, flips, coarse dropout) + intensity transforms (gamma, CLAHE,
      brightness/contrast, Gaussian noise).  Applied to image AND mask jointly.
    - TorchIO: MRI-specific artefacts (random bias field, noise, blur, ghosting).
      Applied to the image only; mask is unaffected.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple


class HippocampusSliceDataset(Dataset):
    """2D slice dataset for hippocampus segmentation with rich augmentation."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        augment: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            data_dir : path to processed data directory containing .npy files
            split    : one of 'train', 'val', 'test'
            augment  : enable Albumentations + TorchIO augmentation pipeline
            seed     : numpy random seed (used for legacy fallback; Albumentations
                       uses its own internal RNG)
        """
        data_dir = Path(data_dir)
        self.images = np.load(data_dir / f"{split}_images.npy")  # (N, H, W)
        self.labels = np.load(data_dir / f"{split}_labels.npy")  # (N, H, W)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        assert len(self.images) == len(self.labels), (
            f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"
        )

        # Lazily import augmentation module so that environments without
        # albumentations/torchio can still load the dataset without augment=True.
        self._aug_fn = None
        if augment:
            try:
                from src.augmentations import apply_augmentation
                self._aug_fn = apply_augmentation
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from src.augmentations import apply_augmentation
                self._aug_fn = apply_augmentation

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].astype(np.float32)  # (H, W)
        lbl = self.labels[idx].astype(np.float32)  # (H, W)

        if self.augment and self._aug_fn is not None:
            # Full Albumentations + TorchIO pipeline
            img, lbl = self._aug_fn(img, lbl)
        elif self.augment:
            # Fallback: manual flips (if augmentation libs not installed)
            if self.rng.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                lbl = np.flip(lbl, axis=1).copy()
            if self.rng.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
                lbl = np.flip(lbl, axis=0).copy()

        # Binarise mask after augmentation (interpolation can create fringe values)
        lbl = (lbl > 0.5).astype(np.float32)

        # (H, W) -> (1, H, W)
        img_t = torch.from_numpy(img).unsqueeze(0)
        lbl_t = torch.from_numpy(lbl).unsqueeze(0)

        return img_t, lbl_t


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    augment: bool = True,
):
    """Create train, val, test DataLoaders.

    Args:
        data_dir   : path to processed data directory
        batch_size : samples per mini-batch
        num_workers: DataLoader worker processes (0 = main process)
        seed       : random seed for DataLoader shuffling
        augment    : whether to apply augmentation to the training split
    """
    train_ds = HippocampusSliceDataset(data_dir, "train",
                                        augment=augment, seed=seed)
    val_ds   = HippocampusSliceDataset(data_dir, "val",   augment=False, seed=seed)
    test_ds  = HippocampusSliceDataset(data_dir, "test",  augment=False, seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
