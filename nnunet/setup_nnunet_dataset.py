"""
Setup nnUNet Dataset004_Hippocampus from raw Task04 data.
Copies and renames files to nnUNet v2 format, creates dataset.json.
"""

import os
import json
import shutil
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
RAW_DIR = Path(r"e:\Self repairing segmentation\data\raw\Task04_Hippocampus")
NNUNET_RAW = Path(r"e:\Self repairing segmentation\data\nnunet_raw\Dataset004_Hippocampus")

IMAGES_TR_SRC = RAW_DIR / "imagesTr"
LABELS_TR_SRC = RAW_DIR / "labelsTr"
IMAGES_TS_SRC = RAW_DIR / "imagesTs"

IMAGES_TR_DST = NNUNET_RAW / "imagesTr"
LABELS_TR_DST = NNUNET_RAW / "labelsTr"
IMAGES_TS_DST = NNUNET_RAW / "imagesTs"

# ─── Copy + rename training images ───────────────────────────────────────────
print("Copying training images...")
count_tr = 0
for f in sorted(IMAGES_TR_SRC.glob("hippocampus_*.nii.gz")):
    # e.g. hippocampus_001.nii.gz  →  hippocampus_001_0000.nii.gz
    stem = f.stem.replace(".nii", "")  # hippocampus_001
    dst_name = stem + "_0000.nii.gz"
    dst = IMAGES_TR_DST / dst_name
    if not dst.exists():
        shutil.copy2(f, dst)
    count_tr += 1
print(f"  {count_tr} training images copied.")

# ─── Copy training labels (no renaming needed) ────────────────────────────────
print("Copying training labels...")
count_lbl = 0
for f in sorted(LABELS_TR_SRC.glob("hippocampus_*.nii.gz")):
    dst = LABELS_TR_DST / f.name
    if not dst.exists():
        shutil.copy2(f, dst)
    count_lbl += 1
print(f"  {count_lbl} training labels copied.")

# ─── Copy + rename test images ────────────────────────────────────────────────
print("Copying test images...")
count_ts = 0
for f in sorted(IMAGES_TS_SRC.glob("hippocampus_*.nii.gz")):
    stem = f.stem.replace(".nii", "")
    dst_name = stem + "_0000.nii.gz"
    dst = IMAGES_TS_DST / dst_name
    if not dst.exists():
        shutil.copy2(f, dst)
    count_ts += 1
print(f"  {count_ts} test images copied.")

# ─── Build training list for dataset.json ────────────────────────────────────
training_cases = []
for f in sorted(LABELS_TR_DST.glob("hippocampus_*.nii.gz")):
    case_id = f.stem.replace(".nii", "")  # hippocampus_001
    training_cases.append(case_id)

# ─── Write nnUNet v2 dataset.json ─────────────────────────────────────────────
dataset_json = {
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "Anterior": 1,
        "Posterior": 2
    },
    "numTraining": len(training_cases),
    "file_ending": ".nii.gz",
    "name": "Hippocampus",
    "description": "Left and right hippocampus segmentation (Medical Segmentation Decathlon Task04)",
    "reference": "Vanderbilt University Medical Center",
    "licence": "CC-BY-SA 4.0",
    "overwrite_image_reader_writer": "SimpleITKIO"
}

out_json = NNUNET_RAW / "dataset.json"
with open(out_json, "w") as fh:
    json.dump(dataset_json, fh, indent=4)
print(f"\ndataset.json written: {out_json}")
print(f"numTraining: {len(training_cases)}")
print(f"Test images: {count_ts}")
print("\n✅ nnUNet dataset setup complete!")
print(f"   Location: {NNUNET_RAW}")

