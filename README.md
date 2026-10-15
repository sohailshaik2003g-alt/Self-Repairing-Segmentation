# Self-Repairing Neural Cellular Automata for 2D Medical Image Segmentation

> **One-day proof-of-concept** investigating whether a Neural Cellular Automata (NCA)
> segmentation model exhibits self-repair properties — i.e., can it recover a correct
> segmentation after its internal cell-state grid is deliberately corrupted mid-inference?

Inspired by [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
(Mordvintsev et al., 2020), applied to 2D hippocampus segmentation from the
[Medical Segmentation Decathlon Task04](http://medicaldecathlon.com/).

## Hardware Target
- **GPU**: NVIDIA GTX 1650 (4 GB VRAM)
- **RAM**: 8–16 GB
- **Time budget**: one working day, from zero to demonstrable results

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & preprocess data (Task04 Hippocampus)
python scripts/download_data.py
python scripts/preprocess_data.py

# 3. Train the NCA segmentation model
python scripts/train.py

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
│   ├── dataset.py          # PyTorch Dataset for preprocessed 2D slices
│   ├── nca_model.py        # NCA segmentation model
│   ├── train_utils.py      # Training loop, losses, metrics
│   ├── damage.py           # Damage/corruption functions
│   └── visualization.py    # Plotting & GIF generation
├── scripts/
│   ├── download_data.py    # Download Task04 hippocampus
│   ├── preprocess_data.py  # Extract 2D slices from 3D NIfTI
│   ├── train.py            # Main training script
│   ├── repair_experiment.py# Self-repair sweep experiment
│   └── visualize.py        # Generate figures & GIFs
├── data/                   # Raw & processed data (gitignored)
├── checkpoints/            # Model checkpoints (gitignored)
├── results/                # Experiment outputs, plots, GIFs
└── docs/
    └── REPORT.md           # Final write-up with real numbers
```

## Key Results Summary
- **Model Size:** 4,352 trainable parameters (trains in ~4.3 min on GTX 1650).
- **Segmentation Performance:** Best Val Dice = **0.5627**, Test Dice = **0.5598**.
- **Self-Repair Recovery:**
  - **10% damage:** **99.3% – 100.5% recovery** (damage completely healed).
  - **30% damage:** **79.1% – 89.7% recovery**.
  - **50% damage:** **74.5% – 86.7% recovery** even after half the grid is destroyed.

Full experimental write-up and analysis: see [docs/REPORT.md](docs/REPORT.md).

## Shortcuts Taken Due to time and hardware constraint
- **2D only**: we extract axial slices from 3D volumes instead of doing full 3D NCA.
- **Small dataset**: 40 volumes (1,044 slices) subset of Task04, not the full dataset.
- **Small image size**: 64×64 to fit in 4 GB VRAM with fast backprop-through-time.
- **No baseline comparison**: no U-Net or other architecture for comparison.
- **Modest accuracy**: we target "clearly segments the hippocampus" (Dice ~0.5–0.7),
  not state-of-the-art clinical results.
- **Simple damage model**: rectangular zero-out regions only.

## License
MIT — educational proof-of-concept.
