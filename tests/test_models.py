"""Tests for model architecture builders — no saved weights or GPU required."""

import numpy as np
import tensorflow as tf

from src.common import AttentionLayer, focal_loss, f1_m


def _build_bilstm_baseline(input_shape, num_classes=3, dropout_rate=0.4, l2_rate=1e-4):
    """Reproduce the lstm_bi architecture from src/training/lstm_model_bi.py."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
    from tensorflow.keras import regularizers

    l2_reg = regularizers.l2(l2_rate)
    model = Sequential()
    model.add(Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg),
        input_shape=input_shape,
    ))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(
        LSTM(64, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg),
    ))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax", kernel_regularizer=l2_reg))
    return model


def _build_forecast(input_shape, num_classes=4, dropout_rate=0.4, l2_rate=1e-4):
    """Reproduce the forecast architecture from src/training/forecast_model.py."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
    from tensorflow.keras import regularizers

    l2_reg = regularizers.l2(l2_rate)
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg))(inp)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = AttentionLayer()(x)
    out = Dense(num_classes, activation="softmax", kernel_regularizer=l2_reg)(x)
    return Model(inputs=inp, outputs=out)


def test_bilstm_baseline_3class():
    model = _build_bilstm_baseline((20, 5), num_classes=3)
    dummy = np.random.randn(2, 20, 5).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_forecast_4class():
    model = _build_forecast((20, 5), num_classes=4)
    dummy = np.random.randn(2, 20, 5).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_forecast_3class_20features():
    model = _build_forecast((20, 20), num_classes=3)
    dummy = np.random.randn(2, 20, 20).astype(np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_model_compile_with_focal_loss():
    """Models should compile successfully with focal_loss and f1_m."""
    from tensorflow.keras.optimizers import Adam

    model = _build_forecast((20, 5), num_classes=4)
    model.compile(
        loss=focal_loss(gamma=2.0, alpha=0.25),
        optimizer=Adam(learning_rate=1e-4),
        metrics=["accuracy", f1_m],
    )
    assert model.loss is not None
