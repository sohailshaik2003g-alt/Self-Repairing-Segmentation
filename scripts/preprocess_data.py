"""
Preprocess Task04 Hippocampus: extract 2D axial slices from 3D NIfTI volumes.

Steps:
  1. Load each 3D volume + label pair.
  2. For each axial slice, check if the label contains hippocampus (>0 pixels).
  3. Skip empty slices, keep slices with hippocampus.
  4. Resize to IMG_SIZE x IMG_SIZE (default 64x64).
  5. Normalize image intensities to [0, 1].
  6. Convert multi-class labels to binary (hippocampus vs background).
  7. Save as numpy arrays, split by patient into train/val/test.

Usage:
    python scripts/preprocess_data.py [--img_size 64] [--max_volumes 40]
"""
import os
import sys
import argparse
import json
import random
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

# Fixed seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Task04_Hippocampus"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def resize_slice(img: np.ndarray, label: np.ndarray, target_size: int):
    """Resize a 2D image and label to target_size x target_size."""
    h, w = img.shape
    zoom_h, zoom_w = target_size / h, target_size / w

    img_resized = zoom(img, (zoom_h, zoom_w), order=1)  # bilinear for image
    label_resized = zoom(label.astype(float), (zoom_h, zoom_w), order=0)  # nearest for label
    label_resized = (label_resized > 0.5).astype(np.uint8)  # binary

    return img_resized, label_resized


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] using min-max."""
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)


def extract_slices_from_volume(
    img_path: Path, lbl_path: Path, target_size: int
):
    """Extract non-empty 2D axial slices from a 3D volume."""
    img_nii = nib.load(str(img_path))
    lbl_nii = nib.load(str(lbl_path))
    img_3d = img_nii.get_fdata().astype(np.float32)
    lbl_3d = lbl_nii.get_fdata().astype(np.uint8)

    slices = []
    n_axial = img_3d.shape[2]  # typically the last axis is axial for hippocampus

    for z in range(n_axial):
        img_slice = img_3d[:, :, z]
        lbl_slice = lbl_3d[:, :, z]

        # Skip slices without any hippocampus
        if lbl_slice.max() == 0:
            continue

        # Binarize label: any hippocampus class > 0 → foreground
        lbl_binary = (lbl_slice > 0).astype(np.uint8)

        # Resize
        img_r, lbl_r = resize_slice(img_slice, lbl_binary, target_size)

        # Normalize image
        img_r = normalize_intensity(img_r)

        slices.append((img_r, lbl_r))

    return slices


def main():
    parser = argparse.ArgumentParser(description="Preprocess Task04 Hippocampus")
    parser.add_argument("--img_size", type=int, default=64, help="Target image size")
    parser.add_argument("--max_volumes", type=int, default=40,
                        help="Maximum number of volumes to process (for speed)")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    img_dir = RAW_DATA_DIR / "imagesTr"
    lbl_dir = RAW_DATA_DIR / "labelsTr"

    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found. Run download_data.py first.")
        sys.exit(1)

    # Gather all volume paths, sorted for reproducibility
    # Filter out macOS resource fork files (._*)
    img_files = sorted([f for f in img_dir.glob("*.nii.gz") if not f.name.startswith("._")])
    lbl_files = sorted([f for f in lbl_dir.glob("*.nii.gz") if not f.name.startswith("._")])

    # Limit to max_volumes
    if args.max_volumes and len(img_files) > args.max_volumes:
        img_files = img_files[: args.max_volumes]
        lbl_files = lbl_files[: args.max_volumes]

    n_vols = len(img_files)
    print(f"Processing {n_vols} volumes at {args.img_size}x{args.img_size} ...")

    # Patient-level split (by volume index)
    indices = list(range(n_vols))
    random.shuffle(indices)
    n_train = int(n_vols * args.train_frac)
    n_val = int(n_vols * args.val_frac)
    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train : n_train + n_val])
    test_idx = set(indices[n_train + n_val :])

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_data = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    split_meta = {"train": [], "val": [], "test": []}  # track provenance

    total_slices = 0
    for i, (img_f, lbl_f) in enumerate(
        tqdm(zip(img_files, lbl_files), total=n_vols, desc="Extracting slices")
    ):
        slices = extract_slices_from_volume(img_f, lbl_f, args.img_size)
        total_slices += len(slices)

        # Determine split
        for split_name, idx_set in splits.items():
            if i in idx_set:
                for img_s, lbl_s in slices:
                    split_data[split_name].append(img_s)
                    split_labels[split_name].append(lbl_s)
                    split_meta[split_name].append(
                        {"volume": img_f.name, "vol_idx": i}
                    )
                break

    # Save as numpy arrays
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split_name in ["train", "val", "test"]:
        if len(split_data[split_name]) == 0:
            print(f"[WARNING] No slices for split '{split_name}'")
            continue
        imgs = np.stack(split_data[split_name], axis=0)  # (N, H, W)
        lbls = np.stack(split_labels[split_name], axis=0)  # (N, H, W)

        np.save(PROCESSED_DIR / f"{split_name}_images.npy", imgs)
        np.save(PROCESSED_DIR / f"{split_name}_labels.npy", lbls)

        stats[split_name] = {
            "n_slices": int(imgs.shape[0]),
            "shape": list(imgs.shape),
            "label_coverage_mean": float(lbls.mean()),
        }
        print(
            f"  {split_name}: {imgs.shape[0]} slices, "
            f"shape={imgs.shape}, label coverage={lbls.mean():.4f}"
        )

    # Save metadata
    meta = {
        "img_size": args.img_size,
        "max_volumes": args.max_volumes,
        "total_volumes_processed": n_vols,
        "total_slices": total_slices,
        "seed": SEED,
        "splits": stats,
    }
    with open(PROCESSED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Total non-empty slices extracted: {total_slices}")
    print(f"  Saved to {PROCESSED_DIR}")
    print(f"  Metadata: {PROCESSED_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
