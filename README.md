# Self-Repairing Neural Cellular Automata for 2D Medical Image Segmentation

> **Proof-of-Concept & Optimization Study** investigating whether a Neural Cellular Automata (NCA)
> segmentation model exhibits self-repair properties — i.e., can it recover a correct
> segmentation after its internal cell-state grid is deliberately corrupted mid-inference?

Inspired by [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
(Mordvintsev et al., 2020), applied to 2D hippocampus segmentation from the
[Medical Segmentation Decathlon Task04](http://medicaldecathlon.com/).

## Hardware Target
- **GPU**: NVIDIA GTX 1650 (4 GB VRAM)
- **RAM**: 8–16 GB

## Quick Start

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

## Project Structure
```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py          # PyTorch Dataset supporting Albumentations + TorchIO
│   ├── augmentations.py    # Combined Albumentations & TorchIO pipeline
│   ├── nca_model.py        # NCA segmentation model
│   ├── train_utils.py      # Training loop, losses, metrics
│   ├── damage.py           # Damage/corruption functions
│   └── visualization.py    # Plotting & GIF generation
├── scripts/
│   ├── download_data.py    # Download Task04 hippocampus
│   ├── preprocess_data.py  # Extract 2D slices from 3D NIfTI
│   ├── train.py            # Main training script (supports --augment)
│   ├── repair_experiment.py# Self-repair sweep experiment
│   └── visualize.py        # Generate figures & GIFs
├── data/                   # Raw & processed data (gitignored)
├── checkpoints/            # Model checkpoints (best_model_aug.pt)
├── results/                # Experiment outputs, plots, GIFs
└── docs/
    └── REPORT.md           # Full experimental report & benchmarks
```

## Benchmark & Performance Progression

| Experiment Phase | Architecture | Augmentation | Parameters | Best Val Dice | **Test Dice** | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 16 ch, 64 hid | None | 4,352 | `0.5627` | `0.5598` | `0.5432` |
| **Augmentation (Short)** | 16 ch, 64 hid | Albu + TorchIO (60 ep) | 4,352 | `0.5676` | `0.5681` | `0.5378` |
| **Augmentation (Extended)** | 16 ch, 64 hid | Albu + TorchIO (187 ep) | 4,352 | `0.6188` | `0.6263` | `0.4768` |
| **Upgraded Capacity (Final)** | **24 ch, 96 hid** | **Albu + TorchIO (180 ep)** | **9,600** | **`0.7475`** | **`0.7173`** | **`0.3538`** |

### Key Results Summary
- **Current Best Validation Dice:** **`0.7475`** (**+18.48%** absolute gain over baseline).
- **Held-Out Test Set Dice:** **`0.7173`** (**+15.75%** absolute gain / **+28.1%** relative improvement).
- **Model Efficiency:** Only **9,600 trainable parameters** (~60 KB checkpoint), training in under 1.9 GB VRAM on a GTX 1650.
- **Self-Repair Properties:**
  - **10% area damage:** **99.3% – 100.5% recovery** (damage completely healed).
  - **30% area damage:** **79.1% – 89.7% recovery**.
  - **50% area damage:** **74.5% – 86.7% recovery** even after half the grid state is destroyed.

Full experimental write-up and analysis: see [docs/REPORT.md](docs/REPORT.md).

## Shortcuts Taken Due to time and hardware constraint
- **2D only**: we extract axial slices from 3D volumes instead of doing full 3D NCA.
- **Small dataset**: 40 volumes (1,044 slices) subset of Task04, not the full dataset.
- **Small image size**: 64×64 to fit in 4 GB VRAM with fast backprop-through-time.
- **No baseline comparison**: no U-Net or other architecture for comparison.
- **Simple damage model**: rectangular zero-out regions only.

## License
MIT — educational proof-of-concept.
