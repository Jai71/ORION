"""Tests for data generation and pipeline utilities — uses dummy data only."""

import numpy as np

from src.data_generation.generate import (
    generate_patient_data,
    build_windows,
    augment_features,
    jitter_windows,
    balance_classes,
    WINDOW,
    TIMESTEPS,
    FEATURES,
)


def test_generate_patient_normal():
    """Normal patient should have label=0, anomaly='none', onset=None."""
    data, label, atype, onset = generate_patient_data(0, timesteps=100, anomaly_type=None)
    assert data.shape == (100, FEATURES)
    assert label == 0
    assert atype == "none"
    assert onset is None


def test_generate_patient_anaphylaxis():
    """Anaphylaxis patient should have label=1 and a valid onset."""
    data, label, atype, onset = generate_patient_data(0, timesteps=100, anomaly_type="anaphylaxis")
    assert data.shape == (100, FEATURES)
    assert label == 1
    assert atype == "anaphylaxis"
    assert onset is not None
    assert 0 <= onset < 100


def test_build_windows_no_horizon():
    """Without horizon, all windows from an anomaly patient share the patient label."""
    np.random.seed(42)
    data = np.random.randn(50, FEATURES)
    data_list = [data]
    anomaly_types = ["anaphylaxis"]
    onsets = [25]

    X, y = build_windows(data_list, anomaly_types, onsets, horizon=0)
    expected_windows = 50 - WINDOW + 1
    assert X.shape == (expected_windows, WINDOW, FEATURES)
    assert y.shape == (expected_windows,)
    # All windows labeled as anaphylaxis (code=1)
    assert all(y == 1)


def test_build_windows_with_horizon():
    """With prediction horizon, only windows near onset should be labeled positive."""
    np.random.seed(42)
    ts = 200
    data = np.random.randn(ts, FEATURES)
    data_list = [data]
    anomaly_types = ["anaphylaxis"]
    onsets = [100]
    horizon = 30

    X, y = build_windows(data_list, anomaly_types, onsets, horizon=horizon)
    # Some windows should be labeled 0 (before onset) and some 1 (near onset)
    assert 0 in y
    assert 1 in y


def test_augment_features_shape():
    """augment_features should produce (WINDOW, 4*FEATURES) from (WINDOW, FEATURES)."""
    window = np.random.randn(WINDOW, FEATURES)
    augmented = augment_features(window)
    assert augmented.shape == (WINDOW, FEATURES * 4)


def test_jitter_windows():
    """Jitter should add augmented copies for anomaly windows only."""
    np.random.seed(42)
    X = np.random.randn(10, WINDOW, FEATURES)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 0])
    X_aug, y_aug = jitter_windows(X, y, k=2, max_shift=3)
    # Original 10 + 2 copies each for 4 anomaly windows = 10 + 8 = 18
    assert len(X_aug) == 18
    assert len(y_aug) == 18


def test_balance_classes():
    """balance_classes should downsample negatives to <= 2x positives."""
    np.random.seed(42)
    X = np.random.randn(100, WINDOW, FEATURES)
    y = np.zeros(100, dtype=int)
    y[90:] = 1  # 10 positives, 90 negatives
    X_bal, y_bal = balance_classes(X, y, num_classes=3)
    n_neg = (y_bal == 0).sum()
    n_pos = (y_bal != 0).sum()
    assert n_neg <= 2 * n_pos
    assert n_pos == 10  # all positives preserved
