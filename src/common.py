"""Shared custom Keras layers, losses, and metrics for ORION models."""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer: focuses on key time-steps in sequence."""

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attn_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attn_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, -1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, -1)
        context = x * alpha
        return K.sum(context, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for multi-class classification.

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Per-class balancing weight(s). Can be a scalar (applied uniformly
            to all classes) or a list/array of per-class weights, e.g. derived
            from inverse class frequencies.
    """
    alpha_t = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha_t * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    return loss


def f1_m(y_true, y_pred):
    """Macro F1 metric for multi-class predictions (per-batch approximation)."""
    y_true = K.cast(y_true, y_pred.dtype)
    # Use argmax -> one_hot instead of round for proper softmax handling
    pred_classes = K.argmax(y_pred, axis=-1)
    y_pred_ = K.one_hot(pred_classes, K.shape(y_pred)[-1])
    y_pred_ = K.cast(y_pred_, y_pred.dtype)
    tp = K.sum(y_true * y_pred_, axis=0)
    fp = K.sum((1 - y_true) * y_pred_, axis=0)
    fn = K.sum(y_true * (1 - y_pred_), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    return K.mean(f1)


CUSTOM_OBJECTS = {
    "AttentionLayer": AttentionLayer,
    "focal_loss": focal_loss,
    "f1_m": f1_m,
}


def validate_array(arr, name="array"):
    """Check for NaN/Inf in a numpy array. Raises ValueError if found."""
    if not np.isfinite(arr).all():
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        raise ValueError(
            f"{name} contains invalid values: {nan_count} NaN, {inf_count} Inf"
        )
