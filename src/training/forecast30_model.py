import numpy as np
import argparse
import os
import sys
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras import regularizers
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common import AttentionLayer, focal_loss, f1_m  # noqa: E402
from src.config import resolve  # noqa: E402


def build_lstm_forecast_model(input_shape, num_classes, dropout_rate=0.4, l2_rate=1e-4):
    """
    Build a forecasting model: Bi-LSTM + Attention for 30s forecasts.
    """
    l2_reg = regularizers.l2(l2_rate)
    inputs = Input(shape=input_shape)
    x = Bidirectional(
        LSTM(128, return_sequences=True,
             kernel_regularizer=l2_reg,
             recurrent_regularizer=l2_reg)
    )(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(
        LSTM(64, return_sequences=True,
             kernel_regularizer=l2_reg,
             recurrent_regularizer=l2_reg)
    )(x)
    x = Dropout(dropout_rate)(x)
    x = AttentionLayer()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2_reg)(x)
    return Model(inputs, outputs)


def _build_balanced_dataset(X_train, y_train, batch_size, augment_positives=False):
    """Build a tf.data pipeline that interleaves positive and negative samples 1:1.

    This addresses the severe class imbalance that causes the default training
    to ignore minority classes entirely (macro F1 = 0.33 without balancing).
    """
    labels_train = np.argmax(y_train, axis=1)
    pos_mask = labels_train != 0
    neg_mask = ~pos_mask
    X_pos, y_pos = X_train[pos_mask], y_train[pos_mask]
    X_neg, y_neg = X_train[neg_mask], y_train[neg_mask]

    pos_ds = tf.data.Dataset.from_tensor_slices((X_pos, y_pos)).shuffle(len(X_pos), seed=42).repeat()
    neg_ds = tf.data.Dataset.from_tensor_slices((X_neg, y_neg)).shuffle(len(X_neg), seed=42).repeat()

    def interleave_fn(p, n):
        return tf.data.Dataset.from_tensors(p).concatenate(tf.data.Dataset.from_tensors(n))

    balanced_ds = tf.data.Dataset.zip((pos_ds, neg_ds)).flat_map(
        lambda p, n: interleave_fn(p, n)
    ).prefetch(tf.data.AUTOTUNE)

    if augment_positives:
        def augment(window, label):
            label_index = tf.argmax(label, axis=-1)
            def apply_aug(x):
                shift = tf.random.uniform([], minval=-3, maxval=4, dtype=tf.int32)
                def shift_right():
                    return tf.concat([x[shift:], tf.repeat(x[-1:], shift, axis=0)], axis=0)
                def shift_left():
                    s = -shift
                    return tf.concat([tf.repeat(x[:1], s, axis=0), x[:-s]], axis=0)
                x_shifted = tf.cond(shift > 0, shift_right,
                                    lambda: tf.cond(shift < 0, shift_left, lambda: x))
                noise = tf.random.normal(shape=tf.shape(x_shifted), mean=0.0, stddev=0.02, dtype=x_shifted.dtype)
                return x_shifted + noise
            window_aug = tf.cond(tf.not_equal(label_index, 0), lambda: apply_aug(window), lambda: window)
            return window_aug, label
        balanced_ds = balanced_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    return balanced_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), len(X_train) // batch_size


def main(args):
    print("🚀 Starting Bi-LSTM+Attention 30s forecast training…")
    X = np.load(args.x_train)  # shape: (N, WINDOW, FEATURES)
    y = np.load(args.y_train)  # shape: (N, num_classes)

    # Compute class weights
    labels = np.argmax(y, axis=1)
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    cw = dict(zip(classes, weights))
    print(f"Using class weights: {cw}")

    # Build model
    input_shape = X.shape[1:]
    num_classes = y.shape[1]
    model = build_lstm_forecast_model(
        input_shape, num_classes,
        dropout_rate=args.dropout_rate,
        l2_rate=args.l2_rate
    )
    optimizer = Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm)
    model.compile(
        loss=focal_loss(gamma=args.focal_gamma, alpha=args.focal_alpha),
        optimizer=optimizer,
        metrics=['accuracy', f1_m]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(args.checkpoint, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=args.log_dir, histogram_freq=1)
    ]

    if args.balanced_sampling:
        split_idx = int((1 - args.validation_split) * X.shape[0])
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        train_ds, steps = _build_balanced_dataset(
            X_train, y_train, args.batch_size, augment_positives=args.augment_positives
        )
        model.fit(
            train_ds,
            epochs=args.epochs,
            steps_per_epoch=steps,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        model.fit(
            X, y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            class_weight=cw,
            callbacks=callbacks,
            verbose=1
        )

    # Save
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)
    print(f"✅ Saved 30s forecast model to {args.output_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Bi-LSTM+Attention 30s forecast model')
    parser.add_argument('--x_train', type=str, default=resolve('data/processed/X_pred30_train.npy'))
    parser.add_argument('--y_train', type=str, default=resolve('data/processed/y_pred30_train.npy'))
    parser.add_argument('--checkpoint', type=str, default=resolve('checkpoints/forecast30_best.h5'))
    parser.add_argument('--output_model', type=str, default=resolve('models/forecast30_model.h5'))
    parser.add_argument('--log_dir', type=str, default=resolve('logs/forecast30'))
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--clipnorm', type=float, default=1.0)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--l2_rate', type=float, default=1e-4)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--balanced_sampling', action='store_true', default=False,
                        help='Use 1:1 pos/neg interleaved sampling to combat class imbalance')
    parser.add_argument('--augment_positives', action='store_true', default=False,
                        help='Apply jitter+noise augmentation on positive samples (requires --balanced_sampling)')
    args = parser.parse_args()
    main(args)
