"""
Self-repair experiment: sweep severities x damage timings.

For each combination, measure Dice of the healed final mask vs. the undamaged
final mask, averaged over test images. Report a table of Dice recovery.

Usage:
    python scripts/repair_experiment.py [--checkpoint checkpoints/best_model.pt]
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import HippocampusSliceDataset
from src.nca_model import NCAModel
from src.damage import create_damage_fn
from src.train_utils import dice_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_dice_between_masks(pred_a: torch.Tensor, pred_b: torch.Tensor,
                                threshold: float = 0.5) -> float:
    """Dice between two sets of logits (both thresholded)."""
    with torch.no_grad():
        a = (torch.sigmoid(pred_a) > threshold).float()
        b = (torch.sigmoid(pred_b) > threshold).float()
        a_flat = a.reshape(a.shape[0], -1)
        b_flat = b.reshape(b.shape[0], -1)
        inter = (a_flat * b_flat).sum(dim=1)
        union = a_flat.sum(dim=1) + b_flat.sum(dim=1)
        dice = (2.0 * inter + 1e-6) / (union + 1e-6)
        return dice.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--n_test_images", type=int, default=50,
                        help="Number of test images to evaluate on")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt["args"]
    model = NCAModel(
        n_channels=model_args["n_channels"],
        hidden_dim=model_args["hidden_dim"],
        fire_rate=model_args["fire_rate"],
        input_channels=1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded model from {} (val dice={:.4f}, epoch {})".format(
        args.checkpoint, ckpt["val_dice"], ckpt["epoch"]))

    # Load test data
    test_ds = HippocampusSliceDataset(args.data_dir, split="test")
    n_test = min(args.n_test_images, len(test_ds))
    print("Using {} test images".format(n_test))

    # Experiment parameters
    severities = [0.10, 0.30, 0.50]
    # Damage timing as fraction of total steps
    timing_fracs = [0.25, 0.50, 0.75]
    damage_steps = [int(f * args.n_steps) for f in timing_fracs]

    results = {}
    damage_seed = 123  # fixed seed for damage location reproducibility

    print("\n=== Self-Repair Experiment ===")
    print("Total NCA steps: {}".format(args.n_steps))
    print("Severities: {}".format(severities))
    print("Damage steps: {} (fractions: {})".format(damage_steps, timing_fracs))

    # First, compute undamaged baseline for all test images
    print("\nComputing undamaged baselines...")
    undamaged_logits_list = []
    gt_labels_list = []

    with torch.no_grad():
        for i in range(n_test):
            img, lbl = test_ds[i]
            img = img.unsqueeze(0).to(device)
            lbl = lbl.unsqueeze(0).to(device)

            logits = model(img, n_steps=args.n_steps)
            undamaged_logits_list.append(logits.cpu())
            gt_labels_list.append(lbl.cpu())

    undamaged_logits = torch.cat(undamaged_logits_list, dim=0)
    gt_labels = torch.cat(gt_labels_list, dim=0)

    baseline_dice = dice_score(undamaged_logits, gt_labels)
    print("Baseline (undamaged) Dice vs GT: {:.4f}".format(baseline_dice))

    # Sweep experiment
    print("\nRunning damage sweep...")
    for sev in severities:
        for d_step, d_frac in zip(damage_steps, timing_fracs):
            key = "sev={:.0%}_step={}({:.0%})".format(sev, d_step, d_frac)
            damage_fn = create_damage_fn(severity=sev, seed=damage_seed)

            healed_logits_list = []
            with torch.no_grad():
                for i in range(n_test):
                    img, lbl = test_ds[i]
                    img = img.unsqueeze(0).to(device)

                    logits, _ = model.forward_with_damage(
                        img, n_steps=args.n_steps,
                        damage_fn=damage_fn, damage_step=d_step,
                        return_all_steps=True,
                    )
                    healed_logits_list.append(logits.cpu())

            healed_logits = torch.cat(healed_logits_list, dim=0)

            # Dice of healed vs GT
            dice_vs_gt = dice_score(healed_logits, gt_labels)
            # Dice of healed vs undamaged (recovery metric)
            dice_vs_undamaged = compute_dice_between_masks(healed_logits, undamaged_logits)

            results[key] = {
                "severity": sev,
                "damage_step": d_step,
                "damage_frac": d_frac,
                "dice_vs_gt": dice_vs_gt,
                "dice_vs_undamaged": dice_vs_undamaged,
                "dice_recovery": dice_vs_gt / max(baseline_dice, 1e-6),
            }

            print("  {} | Dice vs GT={:.4f} | vs undamaged={:.4f} | recovery={:.1%}".format(
                key, dice_vs_gt, dice_vs_undamaged,
                dice_vs_gt / max(baseline_dice, 1e-6)))

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    full_results = {
        "baseline_dice": baseline_dice,
        "n_steps": args.n_steps,
        "n_test_images": n_test,
        "experiments": results,
    }
    with open(str(results_dir / "repair_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SELF-REPAIR RESULTS SUMMARY")
    print("=" * 80)
    print("Baseline Dice (undamaged) vs GT: {:.4f}".format(baseline_dice))
    print()
    print("{:<12} {:<15} {:<15} {:<15} {:<12}".format(
        "Severity", "Damage Timing", "Dice vs GT", "vs Undamaged", "Recovery"))
    print("-" * 69)
    for key, r in results.items():
        print("{:<12} {:<15} {:<15.4f} {:<15.4f} {:<12.1%}".format(
            "{:.0%}".format(r["severity"]),
            "step {} ({:.0%})".format(r["damage_step"], r["damage_frac"]),
            r["dice_vs_gt"],
            r["dice_vs_undamaged"],
            r["dice_recovery"]))
    print("=" * 80)

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for d_frac in timing_fracs:
            sevs = []
            dices_gt = []
            dices_undam = []
            for key, r in results.items():
                if abs(r["damage_frac"] - d_frac) < 0.01:
                    sevs.append(r["severity"])
                    dices_gt.append(r["dice_vs_gt"])
                    dices_undam.append(r["dice_vs_undamaged"])

            label = "damage at {:.0%}".format(d_frac)
            ax1.plot(sevs, dices_gt, "o-", label=label, linewidth=2, markersize=8)
            ax2.plot(sevs, dices_undam, "o-", label=label, linewidth=2, markersize=8)

        ax1.axhline(y=baseline_dice, color="k", linestyle="--",
                     label="Baseline={:.3f}".format(baseline_dice), alpha=0.7)
        ax1.set_xlabel("Damage Severity (fraction of area)")
        ax1.set_ylabel("Dice Score")
        ax1.set_title("Healed Dice vs Ground Truth")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        ax2.set_xlabel("Damage Severity (fraction of area)")
        ax2.set_ylabel("Dice Score")
        ax2.set_title("Healed Dice vs Undamaged Output")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.suptitle("NCA Self-Repair: Dice Recovery After Damage", fontsize=13)
        plt.tight_layout()
        plt.savefig(str(results_dir / "repair_results.png"), dpi=150, bbox_inches="tight")
        print("\nPlot saved to results/repair_results.png")
    except Exception as e:
        print("Could not generate plot:", e)

    print("Results saved to results/repair_results.json")


if __name__ == "__main__":
    main()
