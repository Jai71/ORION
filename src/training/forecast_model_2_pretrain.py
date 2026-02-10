import numpy as np
import argparse
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow as tf
import random

class AttentionLayer(tf.keras.layers.Layer):
    """
    Custom attention layer: focuses on key time-steps in sequence.
    """
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attn_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attn_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch, time_steps, 1)
        e = K.squeeze(e, -1)                  # (batch, time_steps)
        alpha = K.softmax(e)                 # (batch, time_steps)
        alpha = K.expand_dims(alpha, -1)     # (batch, time_steps, 1)
        context = x * alpha                  # (batch, time_steps, features)
        return K.sum(context, axis=1)       # (batch, features)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for multi-class forecasting tasks."""
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    return loss


def f1_m(y_true, y_pred):
    """Macro F1 metric for multi-class predictions."""
    y_true = K.cast(y_true, y_pred.dtype)
    y_pred_ = K.round(y_pred)
    tp = K.sum(y_true * y_pred_, axis=0)
    fp = K.sum((1 - y_true) * y_pred_, axis=0)
    fn = K.sum(y_true * (1 - y_pred_), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    return K.mean(f1)


def build_forecast_model(input_shape, num_classes, dropout_rate=0.4, l2_rate=1e-4):
    """
    Builds a forecasting model to predict one of four classes:
      0: no event, 1: anaphylaxis, 2: malignant hyperthermia, 3: respiratory depression
    Architecture: Bi-LSTM → Dropout → Bi-LSTM → Dropout → Attention → Dense
    """
    l2_reg = regularizers.l2(l2_rate)
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg))(inp)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(64, return_sequences=True,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    attn_out = AttentionLayer()(x)
    out = Dense(num_classes, activation='softmax',
                kernel_regularizer=l2_reg)(attn_out)
    return Model(inputs=inp, outputs=out)


def main(args):
    print("🚀 Starting Bi-LSTM+Attention for 4-class forecasting…")
    X = np.load(args.x_train)          # (N, WINDOW, FEATURES)
    y = np.load(args.y_train)          # (N, num_classes=4)

    labels = np.argmax(y, axis=1)
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    cw = dict(zip(classes, weights))
    print(f"Class weights: {cw}")

    input_shape = X.shape[1:]
    num_classes = y.shape[1]
    model = build_forecast_model(
        input_shape, num_classes,
        dropout_rate=args.dropout_rate,
        l2_rate=args.l2_rate
    )
    opt = Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm)
    model.compile(
        loss=focal_loss(gamma=args.focal_gamma, alpha=args.focal_alpha),
        optimizer=opt,
        metrics=['accuracy', f1_m]
    )

    # --- Balanced per-batch sampling & on-the-fly augmentation ---
    # Manual train/val split
    split_idx = int((1 - args.validation_split) * X.shape[0])
    X_train_full, y_train_full = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    if args.pretrain:
        model.fit(
            X_train_full, y_train_full,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
                ModelCheckpoint(args.checkpoint, monitor='val_loss', save_best_only=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1),
                TensorBoard(log_dir=args.log_dir, histogram_freq=1)
            ],
            verbose=1
        )
    else:
        # Separate positives (event-soon) vs negatives (no-event)
        labels_train = np.argmax(y_train_full, axis=1)
        pos_mask = labels_train != 0
        neg_mask = ~pos_mask
        X_pos, y_pos = X_train_full[pos_mask], y_train_full[pos_mask]
        X_neg, y_neg = X_train_full[neg_mask], y_train_full[neg_mask]

        # Build tf.data datasets
        pos_ds = tf.data.Dataset.from_tensor_slices((X_pos, y_pos)) \
                    .shuffle(len(X_pos), seed=42).repeat()
        neg_ds = tf.data.Dataset.from_tensor_slices((X_neg, y_neg)) \
                    .shuffle(len(X_neg), seed=42).repeat()

        # Interleave one positive and one negative example, then batch
        def interleave_fn(p, n):
            return tf.data.Dataset.from_tensors(p).concatenate(tf.data.Dataset.from_tensors(n))

        balanced_ds = tf.data.Dataset.zip((pos_ds, neg_ds)) \
            .flat_map(lambda p, n: interleave_fn(p, n)) \
            .prefetch(tf.data.AUTOTUNE)

        # Augmentation: jitter and noise on positives
        def augment(window, label):
            # index of the one-hot label (0=no-event, 1-3=event-soon)
            label_index = tf.argmax(label, axis=-1)

            # function to apply jitter + noise
            def apply_aug(x):
                # random shift in [-3,3]
                shift = tf.random.uniform([], minval=-3, maxval=4, dtype=tf.int32)
                # shift right or left, or keep as is
                def shift_right():
                    return tf.concat([x[shift:], tf.repeat(x[-1:], shift, axis=0)], axis=0)
                def shift_left():
                    s = -shift
                    return tf.concat([tf.repeat(x[:1], s, axis=0), x[:-s]], axis=0)
                x_shifted = tf.cond(shift > 0, shift_right,
                                    lambda: tf.cond(shift < 0, shift_left, lambda: x))
                # add Gaussian noise
                noise = tf.random.normal(shape=tf.shape(x_shifted), mean=0.0, stddev=0.02, dtype=x_shifted.dtype)
                return x_shifted + noise

            # apply augmentation only for event-soon windows
            window_aug = tf.cond(
                tf.not_equal(label_index, 0),
                lambda: apply_aug(window),
                lambda: window
            )
            return window_aug, label

        augmented_ds = balanced_ds \
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(args.batch_size) \
            .prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = len(X_train_full) // args.batch_size

        cbs = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
            ModelCheckpoint(args.checkpoint, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1),
            TensorBoard(log_dir=args.log_dir, histogram_freq=1)
        ]

        model.fit(
            augmented_ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            callbacks=cbs,
            verbose=1
        )

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)
    print(f"✅ Saved forecast model to {args.output_model}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train 4-class early-warning forecasting model')
    p.add_argument('--x_train',      type=str,   default='data/processed/X_pred_train.npy')
    p.add_argument('--y_train',      type=str,   default='data/processed/y_pred_train.npy')
    p.add_argument('--checkpoint',   type=str,   default='checkpoints/forecast_best.h5')
    p.add_argument('--output_model', type=str,   default='models/forecast_model.h5')
    p.add_argument('--log_dir',      type=str,   default='logs/forecast')
    p.add_argument('--learning_rate',type=float, default=1e-4)
    p.add_argument('--clipnorm',     type=float, default=1.0)
    p.add_argument('--dropout_rate', type=float, default=0.4)
    p.add_argument('--l2_rate',      type=float, default=1e-4)
    p.add_argument('--focal_gamma',  type=float, default=2.0)
    p.add_argument('--focal_alpha',  type=float, default=0.25)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--validation_split', type=float, default=0.2)
    p.add_argument('--pretrain',     action='store_true', default=False, help='Toggle simple pretraining mode')
    args = p.parse_args()
    main(args)
