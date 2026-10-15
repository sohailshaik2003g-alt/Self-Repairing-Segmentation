"""
Visualization: step-by-step healing GIFs and grid images.

For 2-3 representative test images, generate:
  1. Step-by-step grid showing: undamaged -> damage -> healing -> final
  2. Animated GIF of the healing process

Usage:
    python scripts/visualize.py [--checkpoint checkpoints/best_model.pt]
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import HippocampusSliceDataset
from src.nca_model import NCAModel
from src.damage import create_damage_fn, get_damage_mask


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_healing_grid(
    model, image, label, n_steps, damage_fn, damage_step,
    device, n_display_steps=8
):
    """Generate a grid of images showing the healing process.

    Returns:
        grid_images: list of (title, H, W) tuples for plotting
        seg_at_steps: list of segmentation masks at key steps
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        img_t = image.unsqueeze(0).to(device)

        # Undamaged run
        undamaged_logits, undamaged_states = model(
            img_t, n_steps=n_steps, return_all_steps=True
        )
        undamaged_seg = torch.sigmoid(undamaged_logits[0, 0]).cpu().numpy()

        # Damaged run
        damaged_logits, damaged_states = model.forward_with_damage(
            img_t, n_steps=n_steps,
            damage_fn=damage_fn, damage_step=damage_step,
            return_all_steps=True,
        )
        healed_seg = torch.sigmoid(damaged_logits[0, 0]).cpu().numpy()

    # Collect key frames from the damaged run
    # States are saved at every step; pick evenly spaced ones after damage
    img_np = image[0].cpu().numpy()
    lbl_np = label[0].cpu().numpy()

    # Get segmentation at key steps
    step_indices = []
    n_post_damage = n_steps - damage_step
    if n_post_damage > 0:
        # A few steps before damage, the damage step, then healing steps
        pre_steps = [max(0, damage_step - 2), damage_step - 1] if damage_step > 1 else [0]
        post_steps_idx = np.linspace(damage_step, len(damaged_states) - 1,
                                      min(n_display_steps, n_post_damage), dtype=int).tolist()
        step_indices = pre_steps + post_steps_idx
    else:
        step_indices = list(range(min(len(damaged_states), n_display_steps)))

    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in step_indices:
        if idx not in seen and idx < len(damaged_states):
            seen.add(idx)
            unique_indices.append(idx)
    step_indices = unique_indices

    frames = []
    for si in step_indices:
        seg = torch.sigmoid(damaged_states[si][0, 0:1]).cpu().numpy()[0]
        frames.append((si, seg))

    return {
        "image": img_np,
        "label": lbl_np,
        "undamaged_seg": undamaged_seg,
        "healed_seg": healed_seg,
        "frames": frames,
        "damage_step": damage_step,
        "n_steps": n_steps,
    }


def save_healing_grid_figure(data, severity, save_path):
    """Save a grid figure showing the healing process."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    frames = data["frames"]
    n_frames = len(frames)
    n_cols = min(n_frames + 3, 12)  # image, GT, undamaged, + frames

    fig, axes = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 5))

    # Row 0: image context
    # Col 0: input image
    axes[0, 0].imshow(data["image"], cmap="gray")
    axes[0, 0].set_title("Input", fontsize=8)
    axes[0, 0].axis("off")

    # Col 1: ground truth
    axes[0, 1].imshow(data["label"], cmap="Reds", vmin=0, vmax=1)
    axes[0, 1].set_title("GT Mask", fontsize=8)
    axes[0, 1].axis("off")

    # Col 2: undamaged prediction
    axes[0, 2].imshow(data["undamaged_seg"], cmap="Reds", vmin=0, vmax=1)
    axes[0, 2].set_title("Undamaged", fontsize=8)
    axes[0, 2].axis("off")

    # Remaining cols: healing frames
    for i in range(3, n_cols):
        axes[0, i].axis("off")

    # Row 1: healing timeline
    for col_idx, (step_idx, seg) in enumerate(frames):
        if col_idx >= n_cols:
            break
        ax = axes[1, col_idx]
        ax.imshow(data["image"], cmap="gray", alpha=0.5)
        ax.imshow(seg, cmap="Reds", vmin=0, vmax=1, alpha=0.6)

        is_damage = step_idx == data["damage_step"]
        color = "red" if is_damage else "black"
        prefix = "DMG " if is_damage else ""
        ax.set_title("{}Step {}".format(prefix, step_idx), fontsize=7, color=color)
        ax.axis("off")

    # Hide unused axes
    for col_idx in range(len(frames), n_cols):
        axes[1, col_idx].axis("off")

    plt.suptitle("NCA Healing (severity={:.0%}, damage at step {}/{})".format(
        severity, data["damage_step"], data["n_steps"]), fontsize=11)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def save_healing_gif(data, save_path, fps=4):
    """Save an animated GIF of the healing process."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    frames_data = data["frames"]
    gif_frames = []

    for step_idx, seg in frames_data:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(data["image"], cmap="gray", alpha=0.5)
        ax.imshow(seg, cmap="Reds", vmin=0, vmax=1, alpha=0.6)

        is_damage = step_idx == data["damage_step"]
        color = "red" if is_damage else "black"
        prefix = "DAMAGED! " if is_damage else ""
        ax.set_title("{}Step {}/{}".format(prefix, step_idx, data["n_steps"]),
                     fontsize=10, color=color, fontweight="bold" if is_damage else "normal")
        ax.axis("off")
        plt.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        gif_frames.append(rgb.copy())
        plt.close()

    # Save GIF
    imageio.mimsave(str(save_path), gif_frames, fps=fps, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--severity", type=float, default=0.30)
    parser.add_argument("--damage_frac", type=float, default=0.50,
                        help="Fraction of steps at which to apply damage")
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

    # Load test data
    test_ds = HippocampusSliceDataset(args.data_dir, split="test")

    # Pick examples with good label coverage
    coverages = []
    for i in range(len(test_ds)):
        _, lbl = test_ds[i]
        coverages.append(lbl.mean().item())
    coverages = np.array(coverages)

    # Pick examples from medium-to-high coverage
    sorted_idx = np.argsort(coverages)
    top_half = sorted_idx[len(sorted_idx) // 2:]
    pick_idx = top_half[np.linspace(0, len(top_half) - 1, args.n_examples, dtype=int)]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    damage_step = int(args.damage_frac * args.n_steps)
    damage_fn = create_damage_fn(severity=args.severity, seed=42)

    print("Generating visualizations for {} examples...".format(args.n_examples))
    print("Severity: {:.0%}, Damage at step {}/{}".format(
        args.severity, damage_step, args.n_steps))

    for ex_num, idx in enumerate(pick_idx):
        print("  Example {} (slice #{}, coverage={:.1%})".format(
            ex_num + 1, idx, coverages[idx]))

        image, label = test_ds[idx]
        data = make_healing_grid(
            model, image, label, args.n_steps,
            damage_fn, damage_step, device
        )

        # Save grid figure
        grid_path = results_dir / "healing_grid_ex{}.png".format(ex_num + 1)
        save_healing_grid_figure(data, args.severity, grid_path)
        print("    Saved grid: {}".format(grid_path))

        # Save GIF
        gif_path = results_dir / "healing_gif_ex{}.gif".format(ex_num + 1)
        save_healing_gif(data, gif_path)
        print("    Saved GIF: {}".format(gif_path))

    # Also generate multi-severity comparison for one example
    print("\nGenerating severity comparison...")
    best_idx = pick_idx[0]
    image, label = test_ds[best_idx]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    severities = [0.10, 0.30, 0.50]
    fig, axes = plt.subplots(len(severities), 8, figsize=(20, 3 * len(severities)))

    for row, sev in enumerate(severities):
        d_fn = create_damage_fn(severity=sev, seed=42)
        data = make_healing_grid(model, image, label, args.n_steps,
                                  d_fn, damage_step, device, n_display_steps=6)

        # Col 0: input
        axes[row, 0].imshow(data["image"], cmap="gray")
        axes[row, 0].set_title("Input" if row == 0 else "", fontsize=8)
        axes[row, 0].set_ylabel("sev={:.0%}".format(sev), fontsize=9)
        axes[row, 0].axis("off")

        # Col 1: undamaged
        axes[row, 1].imshow(data["undamaged_seg"], cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_title("Undamaged" if row == 0 else "", fontsize=8)
        axes[row, 1].axis("off")

        # Cols 2-7: healing frames
        for col, (step_idx, seg) in enumerate(data["frames"][:6]):
            ax = axes[row, col + 2]
            ax.imshow(data["image"], cmap="gray", alpha=0.5)
            ax.imshow(seg, cmap="Reds", vmin=0, vmax=1, alpha=0.6)
            is_dmg = step_idx == data["damage_step"]
            color = "red" if is_dmg else "black"
            ax.set_title(("DMG " if is_dmg else "") + "s={}".format(step_idx),
                         fontsize=7, color=color)
            ax.axis("off")

        # Hide unused
        for col in range(len(data["frames"][:6]) + 2, 8):
            axes[row, col].axis("off")

    plt.suptitle("NCA Self-Repair: Severity Comparison (damage at step {}/{})".format(
        damage_step, args.n_steps), fontsize=12)
    plt.tight_layout()
    plt.savefig(str(results_dir / "severity_comparison.png"), dpi=150, bbox_inches="tight")
    print("Saved severity comparison to results/severity_comparison.png")

    print("\nDone! Visualizations saved to results/")


if __name__ == "__main__":
    main()
