"""
Download the Medical Segmentation Decathlon Task04 (Hippocampus) dataset.

The dataset is a ~28 MB tar.gz file containing 3D NIfTI volumes with
hippocampus segmentation labels.

Usage:
    python scripts/download_data.py
"""
import os
import sys
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

# Fixed seed for reproducibility
import random
import numpy as np
random.seed(42)
np.random.seed(42)

# ---------- configuration ----------
# Google Drive hosting of MSD datasets (mirror links)
# Primary: direct download from MSD mirror
DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
DATASET_NAME = "Task04_Hippocampus"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists.")
        return

    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    print(f"Saved to {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """Extract a .tar or .tar.gz file."""
    marker = dest_dir / DATASET_NAME / ".extracted"
    if marker.exists():
        print(f"[skip] Already extracted in {dest_dir / DATASET_NAME}")
        return

    print(f"Extracting {tar_path} -> {dest_dir} ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print("Done extracting.")


def main():
    tar_path = RAW_DATA_DIR / f"{DATASET_NAME}.tar"
    download_file(DATASET_URL, tar_path)
    extract_tar(tar_path, RAW_DATA_DIR)

    # Quick sanity check
    img_dir = RAW_DATA_DIR / DATASET_NAME / "imagesTr"
    lbl_dir = RAW_DATA_DIR / DATASET_NAME / "labelsTr"
    if img_dir.exists():
        n_img = len(list(img_dir.glob("*.nii.gz")))
        n_lbl = len(list(lbl_dir.glob("*.nii.gz")))
        print(f"\n✓ Found {n_img} training images, {n_lbl} training labels.")
    else:
        print(f"[WARNING] Expected directory {img_dir} not found — check extraction.")
        sys.exit(1)


if __name__ == "__main__":
    main()
