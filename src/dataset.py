"""
PyTorch Dataset for preprocessed 2D hippocampus slices.

Loads numpy arrays saved by preprocess_data.py and serves (image, label) pairs
as tensors with shapes:
    image: (1, H, W) float32 in [0, 1]
    label: (1, H, W) float32 in {0, 1}

Optional data augmentation: random horizontal flip, random rotation (small).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple


class HippocampusSliceDataset(Dataset):
    """2D slice dataset for hippocampus segmentation."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        augment: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: path to processed data directory containing .npy files
            split: one of 'train', 'val', 'test'
            augment: whether to apply random augmentations (train only)
            seed: random seed for augmentation reproducibility
        """
        data_dir = Path(data_dir)
        self.images = np.load(data_dir / f"{split}_images.npy")  # (N, H, W)
        self.labels = np.load(data_dir / f"{split}_labels.npy")  # (N, H, W)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        assert len(self.images) == len(self.labels), (
            f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].astype(np.float32)  # (H, W)
        lbl = self.labels[idx].astype(np.float32)  # (H, W)

        if self.augment:
            # Random horizontal flip
            if self.rng.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                lbl = np.flip(lbl, axis=1).copy()
            # Random vertical flip
            if self.rng.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
                lbl = np.flip(lbl, axis=0).copy()

        # Add channel dimension: (H, W) -> (1, H, W)
        img_t = torch.from_numpy(img).unsqueeze(0)
        lbl_t = torch.from_numpy(lbl).unsqueeze(0)

        return img_t, lbl_t


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create train, val, test DataLoaders."""
    from torch.utils.data import DataLoader

    train_ds = HippocampusSliceDataset(data_dir, "train", augment=True, seed=seed)
    val_ds = HippocampusSliceDataset(data_dir, "val", augment=False, seed=seed)
    test_ds = HippocampusSliceDataset(data_dir, "test", augment=False, seed=seed)

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
