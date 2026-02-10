#!/usr/bin/env python3
"""
Consolidated data generation for ORION.

Replaces all 8 individual data_generation_*.py scripts with a single
parameterized script supporting CLI arguments and preset aliases.

Usage:
  python src/data_generation/generate.py --preset lstm_bi
  python src/data_generation/generate.py --preset forecast
  python src/data_generation/generate.py --preset forecast30
  python src/data_generation/generate.py --preset pretrain
  python src/data_generation/generate.py --horizon 60 --num-classes 4 --augment-features
"""

import argparse
import os
import sys
import random

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import resolve  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_PATIENTS = 500
TIMESTEPS = 3000
FEATURES = 5
ANOMALY_PROBABILITY = 0.3
ANOMALY_TYPES = ["anaphylaxis", "malignant_hyperthermia", "respiratory_depression"]
TYPE_TO_CODE = {
    "none": 0,
    "anaphylaxis": 1,
    "malignant_hyperthermia": 2,
    "respiratory_depression": 3,
}
WINDOW = 20

# ---------------------------------------------------------------------------
# Presets (map each model variant to the right flag combination)
# ---------------------------------------------------------------------------
PRESETS = {
    "lstm_bi": {
        "horizon": 0,
        "num_classes": 3,
        "augment_features": False,
        "jitter": False,
        "hard_negatives": False,
        "balance": False,
        "onset_range": "narrow",
        "output_prefix": "",
        "output_format": "npy",
    },
    "forecast": {
        "horizon": 60,
        "num_classes": 4,
        "augment_features": True,
        "jitter": False,
        "hard_negatives": True,
        "balance": False,
        "onset_range": "wide",
        "output_prefix": "pred_",
        "output_format": "npy",
    },
    "forecast30": {
        "horizon": 30,
        "num_classes": 3,
        "augment_features": False,
        "jitter": False,
        "hard_negatives": False,
        "balance": False,
        "onset_range": "narrow",
        "output_prefix": "pred30_",
        "output_format": "npy",
    },
    "pretrain": {
        "horizon": 60,
        "num_classes": 4,
        "augment_features": True,
        "jitter": False,
        "hard_negatives": True,
        "balance": True,
        "onset_range": "wide",
        "output_prefix": "pretrain_",  # saves as X_pretrain.npy
        "output_format": "npy",
    },
}


# ---------------------------------------------------------------------------
# Feature augmentation
# ---------------------------------------------------------------------------
def augment_features(window):
    """Extend (WINDOW, 5) -> (WINDOW, 20) with deltas, rolling mean, rolling std."""
    deltas = np.vstack([np.zeros((1, window.shape[1])), np.diff(window, axis=0)])
    means = np.array([window[max(0, i - 4) : i + 1].mean(axis=0) for i in range(window.shape[0])])
    stds = np.array([window[max(0, i - 4) : i + 1].std(axis=0) for i in range(window.shape[0])])
    return np.concatenate([window, deltas, means, stds], axis=1)


# ---------------------------------------------------------------------------
# Patient generation (identical to all existing scripts)
# ---------------------------------------------------------------------------
def generate_patient_data(patient_id, timesteps=TIMESTEPS, anomaly_type=None,
                          onset_range="narrow", hard_negatives=False):
    """Generate vital-sign time-series for one patient.

    Returns (data, label, anomaly_used, onset_index_or_None).
    """
    baseline_hr = random.uniform(65, 85)
    baseline_bp = random.uniform(110, 130)
    baseline_spo2 = random.uniform(96, 100)
    baseline_rr = random.uniform(12, 16)
    baseline_temp = random.uniform(36.5, 37.5)

    time_series = []
    for t in range(timesteps):
        hr = baseline_hr + np.random.normal(0, 1)
        bp = baseline_bp + np.random.normal(0, 2)
        spo2 = baseline_spo2 + np.random.normal(0, 0.5)
        rr = baseline_rr + np.random.normal(0, 0.5)
        temp = baseline_temp + np.random.normal(0, 0.2)
        time_series.append([hr, bp, spo2, rr, temp])

    label = 0
    anomaly_used = "none"
    onset = None

    # Hard negatives: 5% of normal patients get a brief benign BP dip
    if anomaly_type is None and hard_negatives and random.random() < 0.05:
        dip_start = random.randint(timesteps // 5, timesteps // 2)
        dip_duration = random.randint(30, 120)
        dip_amount = random.uniform(5, 15)
        for t in range(dip_start, min(dip_start + dip_duration, timesteps)):
            ratio = min((t - dip_start) / max(dip_duration / 2, 1), 1.0)
            if t - dip_start > dip_duration / 2:
                ratio = max(1.0 - (t - dip_start - dip_duration / 2) / max(dip_duration / 2, 1), 0)
            time_series[t][1] = baseline_bp - dip_amount * ratio + np.random.normal(0, 2)

    if anomaly_type is not None:
        label = 1
        anomaly_used = anomaly_type
        if onset_range == "wide":
            onset = random.randint(timesteps // 10, int(timesteps * 0.9))
        else:
            onset = random.randint(timesteps // 5, timesteps // 2)

        if anomaly_type == "anaphylaxis":
            drop_onset = random.uniform(0.20, 0.40)
            spo2_drop_onset = random.uniform(0.05, 0.10)
            delta_temp_onset = random.uniform(0.1, 0.2)
            incr_rr_onset = random.uniform(0.20, 0.40)
            incr_rr_overall = random.uniform(0.30, 0.60)
            overall_hr_factor = random.uniform(1.30, 1.50)
            overall_bp_drop = drop_onset + random.uniform(0, 0.05)
            overall_spo2_drop = random.uniform(0.10, 0.15)
            delta_temp2 = random.uniform(0, 0.1)
            for t in range(onset, timesteps):
                dt = t - onset
                if dt <= 300:
                    r = dt / 300.0
                    new_hr = baseline_hr * (1 + 0.30 * r)
                    new_bp = baseline_bp * (1 - drop_onset * r)
                    new_spo2 = baseline_spo2 * (1 - spo2_drop_onset * r)
                    new_temp = baseline_temp + delta_temp_onset * r
                    new_rr = baseline_rr * (1 + incr_rr_onset * r)
                elif dt <= 600:
                    r = (dt - 300) / 300.0
                    new_hr = baseline_hr * (1.30 + (overall_hr_factor - 1.30) * r)
                    new_bp = baseline_bp * ((1 - drop_onset) - ((overall_bp_drop - drop_onset) * r))
                    new_spo2 = baseline_spo2 * ((1 - spo2_drop_onset) - ((overall_spo2_drop - spo2_drop_onset) * r))
                    new_temp = baseline_temp + delta_temp_onset + delta_temp2 * r
                    new_rr = baseline_rr * (1 + incr_rr_onset + (incr_rr_overall - incr_rr_onset) * r)
                else:
                    new_hr = baseline_hr * overall_hr_factor
                    new_bp = baseline_bp * (1 - overall_bp_drop)
                    new_spo2 = baseline_spo2 * (1 - overall_spo2_drop)
                    new_temp = baseline_temp + delta_temp_onset + delta_temp2
                    new_rr = baseline_rr * (1 + incr_rr_overall)
                time_series[t] = [
                    new_hr + np.random.normal(0, 1),
                    new_bp + np.random.normal(0, 2),
                    new_spo2 + np.random.normal(0, 0.5),
                    new_rr + np.random.normal(0, 0.5),
                    new_temp + np.random.normal(0, 0.2),
                ]

        elif anomaly_type == "malignant_hyperthermia":
            drop_onset = random.uniform(0.10, 0.20)
            spO2_drop_onset = random.uniform(0.05, 0.07)
            spO2_drop_total = random.uniform(0.10, 0.15)
            delta_temp1 = random.uniform(0.5, 1.0)
            delta_temp2 = random.uniform(1.0, 1.5)
            rr_onset = random.uniform(0.15, 0.40)
            rr_total = random.uniform(0.40, 0.60)
            for t in range(onset, timesteps):
                dt = t - onset
                if dt <= 300:
                    r = dt / 300.0
                    new_hr = baseline_hr * (1 + 0.4 * r)
                    new_bp = baseline_bp * (1 - drop_onset * r)
                    new_spo2 = baseline_spo2 * (1 - spO2_drop_onset * r)
                    new_temp = baseline_temp + delta_temp1 * r
                    new_rr = baseline_rr * (1 + rr_onset * r)
                elif dt <= 600:
                    r = (dt - 300) / 300.0
                    new_hr = baseline_hr * (1.4 + 0.2 * r)
                    new_bp = baseline_bp * (1 - drop_onset)
                    new_spo2 = baseline_spo2 * (1 - (spO2_drop_onset + (spO2_drop_total - spO2_drop_onset) * r))
                    new_temp = baseline_temp + delta_temp1 + delta_temp2 * r
                    new_rr = baseline_rr * (1 + rr_onset + (rr_total - rr_onset) * r)
                else:
                    new_hr = baseline_hr * 1.6
                    new_bp = baseline_bp * (1 - drop_onset)
                    new_spo2 = baseline_spo2 * (1 - spO2_drop_total)
                    new_temp = baseline_temp + delta_temp1 + delta_temp2
                    new_rr = baseline_rr * (1 + rr_total)
                time_series[t] = [
                    new_hr + np.random.normal(0, 1),
                    new_bp + np.random.normal(0, 2),
                    new_spo2 + np.random.normal(0, 0.5),
                    new_rr + np.random.normal(0, 0.5),
                    new_temp + np.random.normal(0, 0.2),
                ]

        elif anomaly_type == "respiratory_depression":
            hr_drop_onset = random.uniform(0.05, 0.10)
            bp_drop_onset = random.uniform(0.05, 0.10)
            overall_spo2_drop = random.uniform(0.15, 0.20)
            spo2_drop_onset = random.uniform(0.10, 0.13)
            for t in range(onset, timesteps):
                dt = t - onset
                if dt <= 300:
                    r = dt / 300.0
                    new_hr = baseline_hr * (1 - hr_drop_onset * r)
                    new_bp = baseline_bp * (1 - bp_drop_onset * r)
                    spo2_r = 0 if dt < 60 else (dt - 60) / (300 - 60)
                    new_spo2 = baseline_spo2 * (1 - spo2_drop_onset * spo2_r)
                    new_rr = baseline_rr
                    new_temp = baseline_temp
                elif dt <= 600:
                    r = (dt - 300) / 300.0
                    new_hr = baseline_hr * (1 - hr_drop_onset)
                    new_bp = baseline_bp * (1 - bp_drop_onset)
                    new_spo2 = baseline_spo2 * (1 - (spo2_drop_onset + (overall_spo2_drop - spo2_drop_onset) * r))
                    new_rr = baseline_rr
                    new_temp = baseline_temp
                else:
                    new_hr = baseline_hr * (1 - hr_drop_onset)
                    new_bp = baseline_bp * (1 - bp_drop_onset)
                    new_spo2 = baseline_spo2 * (1 - overall_spo2_drop)
                    new_rr = baseline_rr
                    new_temp = baseline_temp
                time_series[t] = [
                    new_hr + np.random.normal(0, 1),
                    new_bp + np.random.normal(0, 2),
                    new_spo2 + np.random.normal(0, 0.5),
                    new_rr + np.random.normal(0, 0.5),
                    new_temp + np.random.normal(0, 0.2),
                ]

    data = np.array(time_series)
    return data, label, anomaly_used, onset


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(num_patients=NUM_PATIENTS, timesteps=TIMESTEPS,
                     onset_range="narrow", hard_negatives=False):
    data_list, labels, anomaly_types, onsets = [], [], [], []
    for pid in range(num_patients):
        atype = random.choice(ANOMALY_TYPES) if random.random() < ANOMALY_PROBABILITY else None
        data, label, aused, onset = generate_patient_data(
            pid, timesteps, atype, onset_range=onset_range, hard_negatives=hard_negatives
        )
        data_list.append(data)
        labels.append(label)
        anomaly_types.append(aused)
        onsets.append(onset)
    return data_list, labels, anomaly_types, onsets


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def build_windows(data_list, anomaly_types, onsets, horizon=0):
    """Create sliding windows with optional prediction-horizon labeling."""
    X_windows, y_codes = [], []
    for series, atype, onset in zip(data_list, anomaly_types, onsets):
        n = len(series)
        if horizon > 0:
            # Prediction-horizon labeling
            for start in range(0, n - WINDOW - horizon + 1):
                window = series[start : start + WINDOW]
                if onset is not None and start + WINDOW <= onset < start + WINDOW + horizon:
                    code = TYPE_TO_CODE[atype]
                else:
                    code = TYPE_TO_CODE["none"]
                X_windows.append(window)
                y_codes.append(code)
        else:
            # Per-patient labeling (all windows from one patient share its label)
            code = TYPE_TO_CODE[atype]
            for start in range(0, n - WINDOW + 1):
                X_windows.append(series[start : start + WINDOW])
                y_codes.append(code)
    return np.stack(X_windows), np.array(y_codes)


# ---------------------------------------------------------------------------
# Jitter augmentation
# ---------------------------------------------------------------------------
def jitter_windows(X, y, k=3, max_shift=5):
    """Create k jittered copies of anomaly windows (shift +-max_shift timesteps)."""
    aug_X, aug_y = [], []
    for i in range(len(X)):
        if y[i] == 0:
            continue
        for _ in range(k):
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                w = np.concatenate([X[i][shift:], np.tile(X[i][-1:], (shift, 1))], axis=0)
            elif shift < 0:
                s = -shift
                w = np.concatenate([np.tile(X[i][:1], (s, 1)), X[i][:-s]], axis=0)
            else:
                w = X[i].copy()
            aug_X.append(w)
            aug_y.append(y[i])
    if aug_X:
        return (
            np.concatenate([X, np.stack(aug_X)]),
            np.concatenate([y, np.array(aug_y)]),
        )
    return X, y


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------
def balance_classes(X, y, num_classes):
    """Downsample majority classes so negative count <= 2x positive count."""
    labels = y if y.ndim == 1 else np.argmax(y, axis=1)
    pos_mask = labels != 0
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    target_neg = min(n_neg, 2 * n_pos)
    neg_indices = np.where(neg_mask)[0]
    np.random.shuffle(neg_indices)
    keep_neg = neg_indices[:target_neg]
    keep = np.sort(np.concatenate([np.where(pos_mask)[0], keep_neg]))
    return X[keep], y[keep]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args):
    np.random.seed(42)
    random.seed(42)

    print(f"Generating data: horizon={args.horizon}, classes={args.num_classes}, "
          f"augment={args.augment_features}, jitter={args.jitter}, "
          f"hard_negatives={args.hard_negatives}, balance={args.balance}")

    # 1. Generate patient data
    data_list, labels, anomaly_types, onsets = generate_dataset(
        onset_range=args.onset_range, hard_negatives=args.hard_negatives
    )
    print(f"Generated {len(data_list)} patients x {TIMESTEPS} timesteps.")

    # 2. Build windows
    X_array, y_codes = build_windows(data_list, anomaly_types, onsets, horizon=args.horizon)
    print(f"Windows: {X_array.shape}, labels: {y_codes.shape}")

    # 3. Optional jitter augmentation
    if args.jitter:
        X_array, y_codes = jitter_windows(X_array, y_codes)
        print(f"After jitter: {X_array.shape}")

    # 4. Optional feature augmentation
    if args.augment_features:
        X_array = np.stack([augment_features(w) for w in X_array])
        print(f"Augmented features: {X_array.shape}")

    # 5. Scale features
    feat_dim = X_array.shape[2]
    flat = X_array.reshape(-1, feat_dim)
    scaler = StandardScaler().fit(flat)
    X_scaled = scaler.transform(flat).reshape(X_array.shape)

    # 6. Filter classes if num_classes == 3 (drop respiratory_depression)
    if args.num_classes == 3:
        mask = y_codes != TYPE_TO_CODE["respiratory_depression"]
        X_scaled = X_scaled[mask]
        y_codes = y_codes[mask]
        print(f"After filtering to 3 classes: {X_scaled.shape}")

    # 7. Optional balancing
    if args.balance:
        X_scaled, y_codes = balance_classes(X_scaled, y_codes, args.num_classes)
        print(f"After balancing: {X_scaled.shape}")

    # 8. One-hot encode
    y_onehot = to_categorical(y_codes, num_classes=args.num_classes)

    # 9. Save
    prefix = args.output_prefix
    out_dir = resolve("data/processed")
    os.makedirs(out_dir, exist_ok=True)

    if args.output_format == "npy":
        x_path = os.path.join(out_dir, f"X_{prefix}train.npy")
        y_path = os.path.join(out_dir, f"y_{prefix}train.npy")
        labels_path = os.path.join(out_dir, f"y_{prefix}labels_full.npy")
        np.save(x_path, X_scaled)
        np.save(y_path, y_onehot)
        np.save(labels_path, y_codes)
        print(f"Saved {x_path} {X_scaled.shape}")
        print(f"Saved {y_path} {y_onehot.shape}")
        print(f"Saved {labels_path} {y_codes.shape}")
    elif args.output_format == "csv":
        import pandas as pd
        # Flatten to per-timestep CSV (no windowing applied for CSV)
        csv_path = os.path.join(resolve("data/raw"), "synthetic_patient_vital_signs.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        rows = []
        for pid, (series, atype) in enumerate(zip(data_list, anomaly_types)):
            for t in range(TIMESTEPS):
                rows.append([pid, t, *series[t], atype])
        df = pd.DataFrame(rows, columns=["patient_id", "timestep", "HR", "BP", "SpO2", "RR", "Temp", "anomaly_type"])
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} ({len(df)} rows)")

    print("Done.")


def build_parser():
    p = argparse.ArgumentParser(description="ORION consolidated data generation")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Use a preset configuration (overrides individual flags)")
    p.add_argument("--horizon", type=int, default=0,
                   help="Prediction horizon in seconds (0=per-patient labeling)")
    p.add_argument("--num-classes", type=int, default=3, choices=[3, 4])
    p.add_argument("--augment-features", action="store_true", default=False)
    p.add_argument("--jitter", action="store_true", default=False)
    p.add_argument("--hard-negatives", action="store_true", default=False)
    p.add_argument("--balance", action="store_true", default=False)
    p.add_argument("--onset-range", choices=["narrow", "wide"], default="narrow")
    p.add_argument("--output-prefix", type=str, default="")
    p.add_argument("--output-format", choices=["npy", "csv"], default="npy")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        args.horizon = preset["horizon"]
        args.num_classes = preset["num_classes"]
        args.augment_features = preset["augment_features"]
        args.jitter = preset["jitter"]
        args.hard_negatives = preset["hard_negatives"]
        args.balance = preset["balance"]
        args.onset_range = preset["onset_range"]
        args.output_prefix = preset["output_prefix"]
        args.output_format = preset["output_format"]

    run(args)


if __name__ == "__main__":
    main()
