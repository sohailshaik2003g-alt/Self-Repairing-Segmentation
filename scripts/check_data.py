"""Quick sanity check of preprocessed data."""
import json
import numpy as np

meta = json.load(open("data/processed/meta.json"))
print("=== Data Pipeline Summary ===")
print("Image size: {}x{}".format(meta["img_size"], meta["img_size"]))
print("Volumes processed:", meta["total_volumes_processed"])
print("Total slices:", meta["total_slices"])
for s, info in meta["splits"].items():
    print("  {}: {} slices, label coverage={:.4f}".format(s, info["n_slices"], info["label_coverage_mean"]))

for split in ["train", "val", "test"]:
    imgs = np.load("data/processed/{}_images.npy".format(split))
    lbls = np.load("data/processed/{}_labels.npy".format(split))
    print("{} images: min={:.3f}, max={:.3f}, dtype={}".format(split, imgs.min(), imgs.max(), imgs.dtype))
    print("{} labels: unique={}, dtype={}".format(split, np.unique(lbls), lbls.dtype))
