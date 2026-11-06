"""
nnUNet Quick Train — single entry point for running nnUNet training.

Sets all required environment variables and launches nnUNetv2_train
directly via python -m (avoids PATH issues on Windows).

Usage:
  python nnunet/run_training.py [--fold 0] [--device cuda]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR     = Path(r"e:\Self repairing segmentation")
DATASET_ID   = "004"
CONFIG       = "2d"

env = os.environ.copy()
env["nnUNet_raw"]          = str(BASE_DIR / "data" / "nnunet_raw")
env["nnUNet_preprocessed"] = str(BASE_DIR / "data" / "nnunet_preprocessed")
env["nnUNet_results"]      = str(BASE_DIR / "data" / "nnunet_results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",   type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Starting nnUNet training: Dataset={DATASET_ID}, Config={CONFIG}, "
          f"Fold={args.fold}, Device={args.device}")
    print(f"nnUNet_raw          = {env['nnUNet_raw']}")
    print(f"nnUNet_preprocessed = {env['nnUNet_preprocessed']}")
    print(f"nnUNet_results      = {env['nnUNet_results']}")
    print()

    cmd = [
        sys.executable, "-m", "nnunetv2.run.run_training",
        DATASET_ID, CONFIG, str(args.fold),
        "--npz",
        "-device", args.device,
    ]
    print(f"CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

