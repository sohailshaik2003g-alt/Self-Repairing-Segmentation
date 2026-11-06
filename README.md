# Hippocampus Segmentation: NCA & nnUNet Approaches

> Two complementary implementations for hippocampus segmentation (Medical Segmentation Decathlon Task04):
>
> 1. **Neural Cellular Automata (NCA)** — a proof-of-concept exploring self-repair properties in segmentation
> 2. **nnUNet** — a state-of-the-art self-configuring medical segmentation framework

Inspired by [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
(Mordvintsev et al., 2020) and [nnU-Net: a self-configuring method](https://www.nature.com/articles/s41592-020-01008-z)
(Isensee et al., 2021), applied to 2D hippocampus segmentation from the
[Medical Segmentation Decathlon Task04](http://medicaldecathlon.com/).

## Hardware Target
- **GPU**: NVIDIA GTX 1650 (4 GB VRAM)
- **RAM**: 8–16 GB

## Quick Start

### NCA Approach
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & preprocess data (Task04 Hippocampus)
python scripts/download_data.py
python scripts/preprocess_data.py

# 3. Train the NCA segmentation model (with Albumentations + TorchIO)
python scripts/train.py --epochs 180 --batch_size 16 --n_channels 24 --hidden_dim 96 --nca_steps_min 20 --nca_steps_max 36 --eval_steps 32 --augment

# 4. Run the self-repair experiment
python scripts/repair_experiment.py

# 5. Generate visualizations
python scripts/visualize.py
```

### nnUNet Approach
```powershell
# 1. Install nnUNet
pip install nnunetv2

# 2. Set environment variables
$env:nnUNet_raw          = "e:\Self repairing segmentation\data\nnunet_raw"
$env:nnUNet_preprocessed = "e:\Self repairing segmentation\data\nnunet_preprocessed"
$env:nnUNet_results      = "e:\Self repairing segmentation\data\nnunet_results"

# 3. Plan & preprocess (already done — skips if outputs exist)
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d 004 -c 2d

# 4. Train (2D config for 4 GB VRAM, fold 0)
python nnunet/run_training.py --fold 0 --device cuda

# 5. Predict + Evaluate
python nnunet/run_predict_eval.py --fold 0
```

## Project Structure
```
├── README.md
├── requirements.txt
├── src/                        # NCA implementation
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset supporting Albumentations + TorchIO
│   ├── augmentations.py        # Combined Albumentations & TorchIO pipeline
│   ├── nca_model.py            # NCA segmentation model
│   ├── train_utils.py          # Training loop, losses, metrics
│   ├── damage.py               # Damage/corruption functions
│   └── visualization.py        # Plotting & GIF generation
├── scripts/                    # NCA pipeline scripts
│   ├── download_data.py        # Download Task04 hippocampus
│   ├── preprocess_data.py      # Extract 2D slices from 3D NIfTI
│   ├── train.py                # Main NCA training script
│   ├── repair_experiment.py    # Self-repair sweep experiment
│   └── visualize.py            # Generate figures & GIFs
├── nnunet/                     # nnUNet implementation
│   ├── README.md               # nnUNet-specific instructions
│   ├── setup_nnunet_dataset.py # One-time dataset setup
│   ├── run_training.py         # Training launcher
│   ├── run_predict_eval.py     # Inference + evaluation
│   ├── train_nnunet.py         # Full pipeline wrapper
│   └── evaluate_predictions.py # Dice evaluation & comparison
├── data/                       # Raw & processed data (gitignored)
├── checkpoints/                # NCA model checkpoints (best_model_aug.pt)
├── results/                    # Experiment outputs, plots, GIFs
└── docs/
    ├── REPORT.md               # Full experimental report & benchmarks
    └── PROJECT_MANAGEMENT_GUIDE.md
```

## Benchmark & Performance

### NCA Results (Neural Cellular Automata)

| Experiment Phase | Architecture | Augmentation | Parameters | Best Val Dice | **Test Dice** | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 16 ch, 64 hid | None | 4,352 | `0.5627` | `0.5598` | `0.5432` |
| **Augmentation (Short)** | 16 ch, 64 hid | Albu + TorchIO (60 ep) | 4,352 | `0.5676` | `0.5681` | `0.5378` |
| **Augmentation (Extended)** | 16 ch, 64 hid | Albu + TorchIO (187 ep) | 4,352 | `0.6188` | `0.6263` | `0.4768` |
| **Upgraded Capacity (Final)** | **24 ch, 96 hid** | **Albu + TorchIO (180 ep)** | **9,600** | **`0.7475`** | **`0.7173`** | **`0.3538`** |

### nnUNet Results (2D Configuration)

| Configuration | VRAM | Val Dice (Mean) | Anterior (Label 1) | Posterior (Label 2) | vs. NCA Baseline |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`2d` (Fold 0)** | **1.58 GB** | **`0.7887`** | **`0.7857`** | **`0.7917`** | **+0.0412 vs Val / +0.0714 vs Test** |

> Metrics verified empirically on held-out 52 validation cases via `results/nnunet_eval/metrics.json`.

### Key Results Summary (NCA)
- **Current Best Validation Dice:** **`0.7475`** (**+18.48%** absolute gain over baseline).
- **Held-Out Test Set Dice:** **`0.7173`** (**+15.75%** absolute gain / **+28.1%** relative improvement).
- **Model Efficiency:** Only **9,600 trainable parameters** (~60 KB checkpoint), training in under 1.9 GB VRAM on a GTX 1650.
- **Self-Repair Properties:**
  - **10% area damage:** **99.3% – 100.5% recovery** (damage completely healed).
  - **30% area damage:** **79.1% – 89.7% recovery**.
  - **50% area damage:** **74.5% – 86.7% recovery** even after half the grid state is destroyed.

Full experimental write-up and analysis: see [docs/REPORT.md](docs/REPORT.md).

Project tracking and documentation steps: see [docs/PROJECT_MANAGEMENT_GUIDE.md](docs/PROJECT_MANAGEMENT_GUIDE.md).

## Shortcuts Taken (NCA)
- **2D only**: we extract axial slices from 3D volumes instead of doing full 3D NCA.
- **Small dataset**: 40 volumes (1,044 slices) subset of Task04, not the full dataset.
- **Small image size**: 64×64 to fit in 4 GB VRAM with fast backprop-through-time.
- **No baseline comparison**: the nnUNet section now provides a strong baseline.
- **Simple damage model**: rectangular zero-out regions only.

## License
MIT — educational proof-of-concept.
