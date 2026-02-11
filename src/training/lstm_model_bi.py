import numpy as np
import argparse
import os
import sys
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras import regularizers
import tensorflow as tf
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common import focal_loss, f1_m  # noqa: E402
from src.config import resolve  # noqa: E402

def build_lstm_model(input_shape, num_classes=3, dropout_rate=0.4, l2_rate=1e-4):
    """
    Build a Bidirectional stacked LSTM model with regularization:
      - Two Bidirectional LSTM layers (128, then 64 units)
      - Dropout between layers
      - L2 kernel+recurrent regularization on all layers
      - Dense output with Softmax
    """
    l2_reg = regularizers.l2(l2_rate)
    model = Sequential()
    # 1st Bidirectional LSTM
    model.add(Bidirectional(
        LSTM(128, return_sequences=True,
             kernel_regularizer=l2_reg,
             recurrent_regularizer=l2_reg),
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))
    # 2nd Bidirectional LSTM
    model.add(Bidirectional(
        LSTM(64,
             kernel_regularizer=l2_reg,
             recurrent_regularizer=l2_reg)
    ))
    model.add(Dropout(dropout_rate))
    # Output layer
    model.add(Dense(
        units=num_classes,
        activation='softmax',
        kernel_regularizer=l2_reg
    ))
    return model

def main(args):
    # Reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    print("🚀 Starting LSTM training…")

    # Load data
    X_train = np.load(args.x_train)
    y_train = np.load(args.y_train)

    # Class weights
    y_labels = np.argmax(y_train, axis=1)
    classes = np.unique(y_labels)
    weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_labels
)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Using class weights: {class_weight_dict}")

    alpha = args.focal_alpha if args.focal_alpha is not None else weights / weights.sum()
    print(f"Focal loss alpha: {alpha}")

    # Build & compile
    input_shape = X_train.shape[1:]  # (time_steps, features)
    num_classes = y_train.shape[1]
    model = build_lstm_model(
        input_shape,
        num_classes,
        dropout_rate=args.dropout_rate,
        l2_rate=args.l2_rate
    )
    optimizer = Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm)
    model.compile(
        loss=focal_loss(gamma=args.focal_gamma, alpha=alpha),
        optimizer=optimizer,
        metrics=['accuracy', f1_m]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=args.checkpoint, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=args.log_dir, histogram_freq=1)
    ]

    # Fit
    model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)
    print(f"✅ Model saved to {args.output_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Bidirectional LSTM anomaly detection model')
    parser.add_argument('--x_train',      type=str,   default=resolve('data/processed/X_train.npy'),    help='Path to X_train file')
    parser.add_argument('--y_train',      type=str,   default=resolve('data/processed/y_train.npy'),    help='Path to y_train file')
    parser.add_argument('--checkpoint',   type=str,   default=resolve('checkpoints/best_lstm.h5'), help='Checkpoint path')
    parser.add_argument('--output_model', type=str,   default=resolve('models/final_lstm.h5'),   help='Final model path')
    parser.add_argument('--log_dir',      type=str,   default=resolve('logs/fit'),       help='TensorBoard log dir')
    parser.add_argument('--learning_rate',type=float, default=1e-4,            help='Initial learning rate')
    parser.add_argument('--clipnorm',     type=float, default=1.0,             help='Gradient clipping norm')
    parser.add_argument('--dropout_rate', type=float, default=0.4,             help='Dropout rate')
    parser.add_argument('--l2_rate',      type=float, default=1e-4,            help='L2 regularization rate')
    parser.add_argument('--focal_gamma',  type=float, default=2.0,             help='Focal loss gamma')
    parser.add_argument('--focal_alpha',  type=float, default=None,
                        help='Uniform focal loss alpha (overrides per-class computation from class frequencies)')
    parser.add_argument('--batch_size',   type=int,   default=32,              help='Batch size')
    parser.add_argument('--epochs',       type=int,   default=50,              help='Max epochs')
    parser.add_argument('--validation_split', type=float, default=0.2,         help='Validation split fraction')
    args = parser.parse_args()
    main(args)