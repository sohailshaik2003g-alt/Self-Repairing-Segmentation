"""
Evaluate nnUNet predictions against ground-truth labels.

Computes per-label and mean Dice scores for test-set predictions.
Saves results to results/nnunet_eval/metrics.json and prints a summary.

Assumes:
  - Predictions in: data/nnunet_predictions/   (*.nii.gz)
  - Ground-truth labels must come from the original imagesTr labels
    for cases that were held out as "test" in our custom split.
  - For the official Task04 test set (imagesTs), ground-truth is NOT
    provided by the decathlon — we therefore evaluate on the validation
    fold that nnUNet creates internally during training (fold 0 val).

This script reads nnUNet's own validation metrics from:
  data/nnunet_results/Dataset004_Hippocampus/
      nnUNetTrainer__nnUNetPlans__2d/
          fold_0/
              validation/
                  summary.json
and re-reports them in a human-readable form alongside the NCA baseline.
"""

import json
from pathlib import Path

BASE_DIR    = Path(r"e:\Self repairing segmentation")
RESULTS_DIR = BASE_DIR / "data" / "nnunet_results"
EVAL_OUT    = BASE_DIR / "results" / "nnunet_eval"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

# ─── Locate nnUNet summary.json ──────────────────────────────────────────────
pattern = "Dataset004_Hippocampus/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation/summary.json"
summary_path = RESULTS_DIR / pattern

if not summary_path.exists():
    # Try without .exe suffix structure — search recursively
    hits = list(RESULTS_DIR.rglob("summary.json"))
    if hits:
        summary_path = hits[0]
        print(f"Found summary at: {summary_path}")
    else:
        print("❌ summary.json not found. Has training completed?")
        print(f"   Expected: {RESULTS_DIR / pattern}")
        raise SystemExit(1)

with open(summary_path) as f:
    data = json.load(f)

# ─── Parse metrics ───────────────────────────────────────────────────────────
# nnUNet v2 summary format:
#   data["metric_per_case"] — list of per-case dicts
#   data["foreground_mean"] — mean over foreground classes
#   data["mean"]            — mean over all classes including background

print("\n" + "=" * 60)
print("nnUNet v2 — Validation Metrics (fold 0, 2D config)")
print("=" * 60)

# Mean Dice across all foreground labels
fg_mean = data.get("foreground_mean", {})
mean_dice = fg_mean.get("Dice", None)

# Per-label means
metrics_mean = data.get("mean", {})

results_to_save = {
    "config": "2d",
    "fold": 0,
    "foreground_mean_dice": mean_dice,
    "per_class": metrics_mean,
}

print(f"\nForeground mean Dice: {mean_dice:.4f}" if mean_dice else "Mean Dice: N/A")
print("\nPer-class metrics:")
for cls_name, cls_metrics in metrics_mean.items():
    dice = cls_metrics.get("Dice", "N/A")
    if isinstance(dice, float):
        print(f"  {cls_name:20s}: Dice = {dice:.4f}")
    else:
        print(f"  {cls_name:20s}: Dice = {dice}")

# ─── NCA Baseline for comparison ─────────────────────────────────────────────
nca_val_dice  = 0.7475
nca_test_dice = 0.7173

print("\n" + "-" * 60)
print("Comparison with NCA baseline:")
print(f"  NCA   Val Dice:  {nca_val_dice:.4f}")
print(f"  NCA   Test Dice: {nca_test_dice:.4f}")
if mean_dice:
    delta = mean_dice - nca_test_dice
    sign  = "+" if delta >= 0 else ""
    print(f"  nnUNet Val Dice: {mean_dice:.4f}  ({sign}{delta:.4f} vs NCA test)")

# ─── Save ────────────────────────────────────────────────────────────────────
out_path = EVAL_OUT / "metrics.json"
with open(out_path, "w") as f:
    json.dump(results_to_save, f, indent=4)
print(f"\n✅ Metrics saved to {out_path}")

