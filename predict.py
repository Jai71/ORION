#!/usr/bin/env python3
"""
Simple inference helper for ORION models.

Examples:
  python3 predict.py --model models/final_lstm.h5 --input data/processed/X_train.npy --sample-index 0
  python3 predict.py --model models/forecast_model.h5 --input data/processed/X_pred_train.npy --sample-index 12 --top-k 4
  python3 predict.py --model models/forecast30_model.h5 --input data/processed/X_pred30_train.npy --sample-index 0 --batch-size 3
"""

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer used by forecast models in this repository."""

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


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Compatibility focal loss for loading trained models."""

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    return loss


def f1_m(y_true, y_pred):
    """Compatibility macro-F1 metric for loading trained models."""
    y_pred_binary = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred_binary, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_binary, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_binary), "float"), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)


def parse_labels(label_arg: Optional[str]) -> Optional[List[str]]:
    """
    Parse labels from either:
    - comma-separated string, or
    - text file path (one label per line).
    """
    if not label_arg:
        return None

    if os.path.isfile(label_arg):
        with open(label_arg, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        labels = [item.strip() for item in label_arg.split(",") if item.strip()]

    return labels or None


def default_labels(num_classes: int) -> List[str]:
    if num_classes == 3:
        return ["none", "anaphylaxis", "malignant_hyperthermia"]
    if num_classes == 4:
        return ["none", "anaphylaxis", "malignant_hyperthermia", "respiratory_depression"]
    return [f"class_{i}" for i in range(num_classes)]


def select_batch(
    array: np.ndarray, sample_index: int, batch_size: int
) -> Tuple[np.ndarray, Sequence[int]]:
    """
    Convert input array into batch form expected by model.

    Supported input shapes:
    - (N, T, F): dataset of windows
    - (T, F): single window
    """
    if array.ndim == 3:
        n = array.shape[0]
        if sample_index < 0 or sample_index >= n:
            raise ValueError(f"--sample-index {sample_index} is out of range for {n} samples")
        if batch_size < 1:
            raise ValueError("--batch-size must be >= 1")
        end = min(sample_index + batch_size, n)
        indices = list(range(sample_index, end))
        return np.asarray(array[sample_index:end]), indices

    if array.ndim == 2:
        if sample_index != 0:
            raise ValueError("Input is a single window (2D), so --sample-index must be 0")
        if batch_size != 1:
            raise ValueError("Input is a single window (2D), so --batch-size must be 1")
        return np.expand_dims(np.asarray(array), axis=0), [0]

    raise ValueError(
        f"Unsupported input shape {array.shape}. Expected (N,T,F) or (T,F)."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference on ORION Keras models using .npy input data."
    )
    parser.add_argument(
        "--model", required=True, help="Path to model file (.h5 or .keras)"
    )
    parser.add_argument(
        "--input", required=True, help="Path to .npy input array (N,T,F) or (T,F)"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Start index for prediction when input is (N,T,F)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of consecutive samples to predict starting at --sample-index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top classes to print per sample",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels or text file path (one label per line)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    custom_objects = {
        "AttentionLayer": AttentionLayer,
        "focal_loss": focal_loss,
        "f1_m": f1_m,
    }
    model = tf.keras.models.load_model(
        args.model, custom_objects=custom_objects, compile=False
    )

    # Memory-map large arrays so only selected slices are materialized in RAM.
    arr = np.load(args.input, mmap_mode="r")
    batch, indices = select_batch(arr, args.sample_index, args.batch_size)

    probs = model.predict(batch, verbose=0)
    if probs.ndim != 2:
        raise ValueError(f"Unexpected model output shape: {probs.shape}")

    num_classes = probs.shape[1]
    labels = parse_labels(args.labels) or default_labels(num_classes)
    if len(labels) != num_classes:
        raise ValueError(
            f"Label count mismatch: got {len(labels)} labels for {num_classes} output classes"
        )

    top_k = max(1, min(args.top_k, num_classes))

    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Selected samples: {indices[0]}..{indices[-1]} ({len(indices)} sample(s))")
    print(f"Input batch shape: {batch.shape}")
    print(f"Output shape: {probs.shape}")
    print("")

    for row_idx, sample_probs in enumerate(probs):
        sample_id = indices[row_idx]
        pred_idx = int(np.argmax(sample_probs))
        pred_label = labels[pred_idx]
        pred_prob = float(sample_probs[pred_idx])

        top_indices = np.argsort(sample_probs)[::-1][:top_k]
        print(
            f"Sample {sample_id}: predicted={pred_label} "
            f"(class={pred_idx}, prob={pred_prob:.6f})"
        )
        print("Top classes:")
        for class_idx in top_indices:
            class_idx = int(class_idx)
            print(f"  - {labels[class_idx]} (class={class_idx}): {float(sample_probs[class_idx]):.6f}")
        print("")


if __name__ == "__main__":
    main()
