#!/usr/bin/env python3
"""
generate_balanced_dataset.py

Creates a fully balanced LSTM training set (equal # windows per class)
using the same simulation + windowing logic as your "real" script.
Saves X_pretrain.npy and y_pretrain.npy for pre-training.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from data_generation_pred60_2 import (
    WINDOW,
    TIMESTEPS,
    TYPE_TO_CODE,
    generate_dataset,
    augment_features,
)

# inverse map code→type (for readability)
CODE_TO_TYPE = {v: k for k, v in TYPE_TO_CODE.items()}

def build_windows(data_list, anomaly_types, onsets, prediction_horizon):
    """Build raw windows and integer labels identical to your real script."""
    X_w, y_codes = [], []
    for series, atype, onset in zip(data_list, anomaly_types, onsets):
        for start in range(0, TIMESTEPS - WINDOW - prediction_horizon + 1):
            window = series[start : start + WINDOW]
            if onset is not None and start + WINDOW <= onset < start + WINDOW + prediction_horizon:
                code = TYPE_TO_CODE[atype]
            else:
                code = TYPE_TO_CODE['none']
            X_w.append(window)
            y_codes.append(code)
    return np.stack(X_w), np.array(y_codes)

def main():
    random.seed(42); np.random.seed(42)

    # 1) Generate the full synthetic cohort
    data_list, labels, anomaly_types, onsets = generate_dataset()

    # 2) Build windows + integer codes (real mix)
    X_raw, y_codes = build_windows(data_list, anomaly_types, onsets, prediction_horizon=60)
    print(f"[1] Raw windows: {X_raw.shape}, labels: {y_codes.shape}")

    # 3) Feature-engineer each window
    X_feat = np.stack([augment_features(w) for w in X_raw])
    n, w, feat_dim = X_feat.shape
    print(f"[2] After augment: {X_feat.shape} (WINDOW, FEATURES*4)")

    # 4) Scale
    flat = X_feat.reshape(-1, feat_dim)
    scaler = StandardScaler().fit(flat)
    X_scaled = scaler.transform(flat).reshape(n, w, feat_dim)

    # 5) Balance classes: undersample majority to match minority
    idx_per_class = {c: np.where(y_codes == c)[0] for c in np.unique(y_codes)}
    counts = {c: len(idx_per_class[c]) for c in idx_per_class}
    print("Class counts before balancing:", counts)

    # target = smallest class size
    target = min(counts.values())
    print("Balancing all classes to:", target, "examples each")

    chosen_idx = []
    for c, idxs in idx_per_class.items():
        if len(idxs) > target:
            chosen = np.random.choice(idxs, size=target, replace=False)
        else:
            # if a class *fewer* than target, oversample with replacement
            chosen = np.random.choice(idxs, size=target, replace=True)
        chosen_idx.append(chosen)
    chosen_idx = np.concatenate(chosen_idx)
    np.random.shuffle(chosen_idx)

    X_bal = X_scaled[chosen_idx]
    y_bal_int = y_codes[chosen_idx]

    # 6) One-hot encode
    num_classes = len(TYPE_TO_CODE)
    y_bal = to_categorical(y_bal_int, num_classes=num_classes)

    # 7) Save
    np.save('data/processed/X_pretrain.npy', X_bal)
    np.save('data/processed/y_pretrain.npy', y_bal)
    print(f"[3] Saved balanced pretrain set: data/processed/X_pretrain.npy {X_bal.shape}, data/processed/y_pretrain.npy {y_bal.shape}")

if __name__ == "__main__":
    main()