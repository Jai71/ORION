"""Tests for src/common.py — shared custom layers, losses, and metrics."""

import numpy as np
import tensorflow as tf

from src.common import AttentionLayer, focal_loss, f1_m, CUSTOM_OBJECTS


def test_attention_layer_output_shape():
    """AttentionLayer should collapse the time axis: (B, T, F) -> (B, F)."""
    layer = AttentionLayer()
    x = tf.random.normal((2, 20, 5))
    out = layer(x)
    assert out.shape == (2, 5)


def test_attention_layer_output_shape_wide():
    """AttentionLayer works with augmented 20-feature input."""
    layer = AttentionLayer()
    x = tf.random.normal((3, 20, 20))
    out = layer(x)
    assert out.shape == (3, 20)


def test_focal_loss_returns_positive_scalar():
    """focal_loss should return a positive loss value."""
    loss_fn = focal_loss(gamma=2.0, alpha=0.25)
    y_true = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], dtype=tf.float32)
    loss_val = loss_fn(y_true, y_pred)
    assert loss_val.shape == (2,)
    assert all(v > 0 for v in loss_val.numpy())


def test_f1_m_returns_value_in_range():
    """f1_m should return a value in [0, 1]."""
    y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=tf.float32)
    result = f1_m(y_true, y_pred).numpy()
    assert 0.0 <= result <= 1.0


def test_f1_m_perfect_predictions():
    """Perfect one-hot predictions should yield F1 close to 1."""
    y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)
    result = f1_m(y_true, y_pred).numpy()
    assert result > 0.9


def test_custom_objects_dict():
    """CUSTOM_OBJECTS should contain all three custom objects."""
    assert "AttentionLayer" in CUSTOM_OBJECTS
    assert "focal_loss" in CUSTOM_OBJECTS
    assert "f1_m" in CUSTOM_OBJECTS
    assert CUSTOM_OBJECTS["AttentionLayer"] is AttentionLayer
    assert CUSTOM_OBJECTS["focal_loss"] is focal_loss
    assert CUSTOM_OBJECTS["f1_m"] is f1_m
