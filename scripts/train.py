"""
Main training script for the NCA segmentation model.

Usage:
    python scripts/train.py [--epochs 100] [--batch_size 8] [--lr 1e-3]

Designed for GTX 1650 (4 GB VRAM): 16 channels, 64x64 images, batch_size=8.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
import random

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import get_dataloaders
from src.nca_model import NCAModel, count_parameters
from src.train_utils import train_one_epoch, evaluate


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train NCA segmentation model")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--n_channels", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--fire_rate", type=float, default=0.5)
    parser.add_argument("--nca_steps_min", type=int, default=32)
    parser.add_argument("--nca_steps_max", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=64,
                        help="Fixed number of steps for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (epochs)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed
    )
    print("Train batches: {}, Val batches: {}, Test batches: {}".format(
        len(train_loader), len(val_loader), len(test_loader)))

    # Model
    model = NCAModel(
        n_channels=args.n_channels,
        hidden_dim=args.hidden_dim,
        fire_rate=args.fire_rate,
        input_channels=1,
    ).to(device)
    print("Model parameters:", count_parameters(model))

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15
    )

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_dice = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    print("\n--- Training ---")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            n_steps_range=(args.nca_steps_min, args.nca_steps_max),
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device, n_steps=args.eval_steps)

        # Record
        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])

        # LR scheduler
        scheduler.step(val_metrics["dice"])

        dt = time.time() - t_epoch
        print("Epoch {:3d}/{} | train loss={:.4f} dice={:.4f} | val loss={:.4f} dice={:.4f} | {:.1f}s".format(
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["dice"],
            val_metrics["loss"], val_metrics["dice"], dt
        ))

        # Checkpointing
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "args": vars(args),
            }, str(ckpt_dir / "best_model.pt"))
            print("  -> New best val dice: {:.4f} (saved)".format(best_val_dice))
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.patience:
            print("Early stopping at epoch {} (no improvement for {} epochs)".format(
                epoch, args.patience))
            break

    total_time = time.time() - t_start
    print("\n--- Training complete ---")
    print("Total time: {:.1f}s ({:.1f} min)".format(total_time, total_time / 60))
    print("Best val Dice: {:.4f}".format(best_val_dice))

    # Final test evaluation with best model
    ckpt = torch.load(str(ckpt_dir / "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, n_steps=args.eval_steps)
    print("Test Dice: {:.4f}, Test Loss: {:.4f}".format(
        test_metrics["dice"], test_metrics["loss"]))

    # Save history
    history["best_val_dice"] = best_val_dice
    history["test_dice"] = test_metrics["dice"]
    history["test_loss"] = test_metrics["loss"]
    history["total_time_s"] = total_time
    history["args"] = vars(args)

    with open(str(ckpt_dir / "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save training curves plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history["train_dice"], label="Train")
        ax2.plot(history["val_dice"], label="Val")
        ax2.axhline(y=test_metrics["dice"], color="r", linestyle="--",
                     label="Test={:.3f}".format(test_metrics["dice"]))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice")
        ax2.set_title("Dice Score")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("NCA Segmentation Training (best val Dice={:.4f})".format(best_val_dice))
        plt.tight_layout()

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        plt.savefig(str(results_dir / "training_curves.png"), dpi=150, bbox_inches="tight")
        print("Training curves saved to results/training_curves.png")
    except Exception as e:
        print("Could not save training curves:", e)

    print("\nDone!")


if __name__ == "__main__":
    main()
