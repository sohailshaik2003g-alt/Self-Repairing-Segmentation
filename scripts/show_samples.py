"""Generate sample images showing a few train slices with their masks."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

imgs = np.load("data/processed/train_images.npy")
lbls = np.load("data/processed/train_labels.npy")

# Pick 6 slices with varying label coverage
coverages = lbls.reshape(len(lbls), -1).mean(axis=1)
# Sort by coverage and pick evenly spaced examples
sorted_idx = np.argsort(coverages)
pick_idx = sorted_idx[np.linspace(0, len(sorted_idx)-1, 6, dtype=int)]

fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for col, idx in enumerate(pick_idx):
    axes[0, col].imshow(imgs[idx], cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title("Slice #{}".format(idx), fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(imgs[idx], cmap="gray", vmin=0, vmax=1, alpha=0.7)
    axes[1, col].imshow(lbls[idx], cmap="Reds", vmin=0, vmax=1, alpha=0.4)
    axes[1, col].set_title("cov={:.1%}".format(coverages[idx]), fontsize=9)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Image", fontsize=10)
axes[1, 0].set_ylabel("+ Mask", fontsize=10)
plt.suptitle("Sample Training Slices (64x64) - Hippocampus Segmentation", fontsize=12)
plt.tight_layout()
plt.savefig(str(out_dir / "sample_slices.png"), dpi=150, bbox_inches="tight")
print("Saved to results/sample_slices.png")
