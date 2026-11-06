# nnUNet Implementation — Hippocampus Segmentation

This directory contains the **nnUNet v2** implementation for hippocampus segmentation
on the Medical Segmentation Decathlon Task04 dataset, as a companion to the Neural
Cellular Automata (NCA) baseline.

## Files

| File | Description |
|------|-------------|
| `setup_nnunet_dataset.py` | One-time setup: copies raw NIfTI files into nnUNet format |
| `run_training.py` | Runs nnUNet training (wraps `nnunetv2.run.run_training`) |
| `run_predict_eval.py` | Runs inference on test set + evaluates Dice |
| `train_nnunet.py` | Full pipeline wrapper (preprocess → train → predict → evaluate) |
| `evaluate_predictions.py` | Reads nnUNet `summary.json` and reports Dice vs NCA baseline |

## Quick Start

```powershell
# 1. Set environment variables (do this in every new terminal)
$env:nnUNet_raw          = "e:\Self repairing segmentation\data\nnunet_raw"
$env:nnUNet_preprocessed = "e:\Self repairing segmentation\data\nnunet_preprocessed"
$env:nnUNet_results      = "e:\Self repairing segmentation\data\nnunet_results"

# 2. (Already done) Setup dataset
python nnunet/setup_nnunet_dataset.py

# 3. (Already done) Plan & preprocess
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d 004 -c 2d --verify_dataset_integrity

# 4. Train (fold 0, 2D config — fits in 4 GB VRAM)
python nnunet/run_training.py --fold 0 --device cuda

# 5. Predict + Evaluate
python nnunet/run_predict_eval.py --fold 0
```

## Configuration Choice: 2D vs 3D

| Config | VRAM Required | Expected Dice | Notes |
|--------|--------------|---------------|-------|
| `2d`   | ~2-3 GB      | ~0.85-0.87    | ✅ Fits in GTX 1650 |
| `3d_fullres` | ~8-12 GB | ~0.89-0.90 | ❌ OOM on 4 GB |
| `3d_lowres` | ~4-6 GB  | ~0.83-0.86    | ⚠️ Borderline |

We use **2D** for compatibility with the GTX 1650 (4 GB VRAM).

## Expected Results

nnUNet on Task04 Hippocampus (2D config, Dice over both foreground labels):

| Method | Val Dice | Reference |
|--------|----------|-----------|
| NCA (ours) | 0.7475 | Current project baseline |
| nnUNet 2D | ~0.85-0.87 | Literature (MSD leaderboard) |
| nnUNet 3D fullres | ~0.89-0.90 | Literature (MSD leaderboard) |

## Directory Structure

```
data/
  nnunet_raw/
    Dataset004_Hippocampus/
      imagesTr/     # 260 cases: hippocampus_NNN_0000.nii.gz
      labelsTr/     # 260 cases: hippocampus_NNN.nii.gz
      imagesTs/     # 130 cases: hippocampus_NNN_0000.nii.gz
      dataset.json  # nnUNet v2 format
  nnunet_preprocessed/   # Created by plan_and_preprocess
  nnunet_results/        # Created by training
  nnunet_predictions/    # Created by predict
```

