#!/usr/bin/env python3
"""Deterministic stratified 80/20 train/test split for all ORION datasets."""

import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import MODEL_REGISTRY, DATA_DIR  # noqa: E402

SEED = 42

DATASETS = [
    {
        "name": name,
        "X": cfg["X_train"],
        "y": cfg["y_train"],
    }
    for name, cfg in MODEL_REGISTRY.items()
]


def split_and_save(name, x_path, y_path, test_size=0.2):
    print(f"--- {name} ---")
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"  Loaded X={X.shape}, y={y.shape}")

    # Stratify on integer class labels
    y_int = np.argmax(y, axis=1) if y.ndim == 2 else y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y_int
    )
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    out_dir = os.path.dirname(x_path)
    for arr, suffix in [
        (X_train, f"X_{name}_split_train.npy"),
        (X_test, f"X_{name}_split_test.npy"),
        (y_train, f"y_{name}_split_train.npy"),
        (y_test, f"y_{name}_split_test.npy"),
    ]:
        path = os.path.join(out_dir, suffix)
        np.save(path, arr)
        print(f"  Saved {path}  {arr.shape}")


def main():
    parser = argparse.ArgumentParser(description="Split ORION datasets 80/20 stratified")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Names to split (default: all). Options: lstm_bi forecast forecast30 pretrain")
    args = parser.parse_args()

    for ds in DATASETS:
        if args.datasets and ds["name"] not in args.datasets:
            continue
        if not os.path.isfile(ds["X"]):
            print(f"Skipping {ds['name']}: {ds['X']} not found")
            continue
        split_and_save(ds["name"], ds["X"], ds["y"], test_size=args.test_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
