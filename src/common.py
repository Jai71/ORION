"""Shared custom Keras layers, losses, and metrics for ORION models."""

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
    """Focal loss for multi-class classification."""

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    return loss


def f1_m(y_true, y_pred):
    """Macro F1 metric for multi-class predictions (type-safe variant)."""
    y_true = K.cast(y_true, y_pred.dtype)
    y_pred_ = K.round(y_pred)
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
