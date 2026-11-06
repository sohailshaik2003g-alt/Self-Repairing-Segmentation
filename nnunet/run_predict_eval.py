"""
nnUNet Predict + Evaluate — runs inference on the test set and evaluates Dice.

Usage:
  python nnunet/run_predict_eval.py [--fold 0]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR   = Path(r"e:\Self repairing segmentation")
DATASET_ID = "004"
CONFIG     = "2d"

env = os.environ.copy()
env["nnUNet_raw"]          = str(BASE_DIR / "data" / "nnunet_raw")
env["nnUNet_preprocessed"] = str(BASE_DIR / "data" / "nnunet_preprocessed")
env["nnUNet_results"]      = str(BASE_DIR / "data" / "nnunet_results")

IMAGES_TS   = BASE_DIR / "data" / "nnunet_raw" / "Dataset004_Hippocampus" / "imagesTs"
PREDICTIONS = BASE_DIR / "data" / "nnunet_predictions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    print(f"Running nnUNet inference: Dataset={DATASET_ID}, Config={CONFIG}, Fold={args.fold}")

    # Predict
    cmd_predict = [
        sys.executable, "-m", "nnunetv2.inference.predict_from_raw_data",
        "-d", DATASET_ID,
        "-i", str(IMAGES_TS),
        "-o", str(PREDICTIONS),
        "-f", str(args.fold),
        "-c", CONFIG,
    ]
    print(f"\nCMD: {' '.join(cmd_predict)}\n")
    result = subprocess.run(cmd_predict, env=env)
    if result.returncode != 0:
        print("Prediction failed.")
        sys.exit(result.returncode)

    # Evaluate
    print("\nRunning evaluation...")
    result2 = subprocess.run(
        [sys.executable, str(BASE_DIR / "nnunet" / "evaluate_predictions.py")],
        env=env
    )
    sys.exit(result2.returncode)


if __name__ == "__main__":
    main()

