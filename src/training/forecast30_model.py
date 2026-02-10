import numpy as np
import argparse
import os
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
import tensorflow.keras.backend as K
import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    """
    Custom attention layer for sequence data.
    """
    def build(self, input_shape):
        self.W = self.add_weight(name='attn_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attn_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x: (batch, time_steps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)       # (batch, time_steps, 1)
        e = K.squeeze(e, -1)                        # (batch, time_steps)
        alpha = K.softmax(e)                        # (batch, time_steps)
        alpha = K.expand_dims(alpha, -1)            # (batch, time_steps, 1)
        context = x * alpha                         # (batch, time_steps, features)
        return K.sum(context, axis=1)              # (batch, features)

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
    """Macro F1 score metric."""
    y_pred_label = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred_label, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_label, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_label), 'float'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)


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


def main(args):
    print("🚀 Starting Bi-LSTM+Attention 30s forecast training…")
    X = np.load(args.x_train)  # shape: (N, WINDOW, FEATURES)
    y = np.load(args.y_train)  # shape: (N, num_classes=4)

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

    # Train
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
    parser.add_argument('--x_train', type=str, default='data/processed/X_pred30_train.npy')
    parser.add_argument('--y_train', type=str, default='data/processed/y_pred30_train.npy')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/forecast30_best.h5')
    parser.add_argument('--output_model', type=str, default='models/forecast30_model.h5')
    parser.add_argument('--log_dir', type=str, default='logs/forecast30')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--clipnorm', type=float, default=1.0)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--l2_rate', type=float, default=1e-4)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--validation_split', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
