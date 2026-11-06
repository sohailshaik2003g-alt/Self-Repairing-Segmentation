"""
nnUNet Training Script for Hippocampus Segmentation (Task04 / Dataset004)
==========================================================================
This script wraps the nnUNet v2 pipeline in a convenient Python interface.
It sets environment variables, then calls the nnUNet CLI tools for:
  1. Plan & Preprocess  (if not already done)
  2. Train              (2D configuration, fold 0)
  3. Predict            (on test set)
  4. Evaluate           (Dice score vs ground truth)

Usage:
  python nnunet/train_nnunet.py [--step {preprocess,train,predict,evaluate,all}]
                                 [--fold FOLD] [--device DEVICE]

Defaults: --step all, --fold 0, --device cuda
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"e:\Self repairing segmentation")
RAW         = BASE_DIR / "data" / "nnunet_raw"
PREPROCESSED= BASE_DIR / "data" / "nnunet_preprocessed"
RESULTS     = BASE_DIR / "data" / "nnunet_results"
SCRIPTS_DIR = Path(r"C:\Users\welcome\AppData\Roaming\Python\Python310\Scripts")

DATASET_ID  = "004"
CONFIG      = "2d"          # 2D for GTX 1650 (4 GB VRAM)

IMAGES_TS   = RAW / "Dataset004_Hippocampus" / "imagesTs"
LABELS_TS   = RAW / "Dataset004_Hippocampus" / "labelsTr"  # used for reference
PREDICTIONS = BASE_DIR / "data" / "nnunet_predictions"
EVAL_OUT    = BASE_DIR / "results" / "nnunet_eval"


def set_env():
    os.environ["nnUNet_raw"]          = str(RAW)
    os.environ["nnUNet_preprocessed"] = str(PREPROCESSED)
    os.environ["nnUNet_results"]      = str(RESULTS)
    # Ensure nnUNet scripts are on PATH
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(SCRIPTS_DIR) not in path_parts:
        os.environ["PATH"] = str(SCRIPTS_DIR) + os.pathsep + os.environ["PATH"]


def run(cmd, **kwargs):
    """Run a shell command and stream output."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)
    return result


def step_preprocess():
    print("=" * 60)
    print("STEP 1: Plan & Preprocess (2D configuration)")
    print("=" * 60)
    run([
        str(SCRIPTS_DIR / "nnUNetv2_plan_and_preprocess.exe"),
        "-d", DATASET_ID,
        "-c", CONFIG,
        "--verify_dataset_integrity",
        "-np", "4",   # preprocessing workers
    ])


def step_train(fold: int = 0, device: str = "cuda"):
    print("=" * 60)
    print(f"STEP 2: Train (config={CONFIG}, fold={fold}, device={device})")
    print("=" * 60)
    run([
        str(SCRIPTS_DIR / "nnUNetv2_train.exe"),
        DATASET_ID, CONFIG, str(fold),
        "--npz",          # save softmax output (needed for ensembling later)
        "-device", device,
    ])


def step_predict(fold: int = 0):
    print("=" * 60)
    print(f"STEP 3: Predict on test set (fold={fold})")
    print("=" * 60)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    run([
        str(SCRIPTS_DIR / "nnUNetv2_predict.exe"),
        "-d", DATASET_ID,
        "-i", str(IMAGES_TS),
        "-o", str(PREDICTIONS),
        "-f", str(fold),
        "-c", CONFIG,
        "--save_probabilities",
    ])


def step_evaluate():
    print("=" * 60)
    print("STEP 4: Evaluate Dice on test predictions")
    print("=" * 60)
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    # nnUNet's built-in evaluator compares predictions vs. reference
    # We evaluate using nnUNetv2_evaluate_folder (needs ground-truth labels)
    # For Task04 test set, labels are NOT provided in imagesTs.
    # So we run our own evaluator (see nnunet/evaluate_predictions.py)
    run([
        sys.executable,
        str(BASE_DIR / "nnunet" / "evaluate_predictions.py"),
    ])


def main():
    parser = argparse.ArgumentParser(description="nnUNet hippocampus pipeline")
    parser.add_argument("--step", default="all",
                        choices=["preprocess", "train", "predict", "evaluate", "all"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_env()

    if args.step in ("preprocess", "all"):
        step_preprocess()
    if args.step in ("train", "all"):
        step_train(fold=args.fold, device=args.device)
    if args.step in ("predict", "all"):
        step_predict(fold=args.fold)
    if args.step in ("evaluate", "all"):
        step_evaluate()

    print("\n✅ nnUNet pipeline finished!")


if __name__ == "__main__":
    main()

