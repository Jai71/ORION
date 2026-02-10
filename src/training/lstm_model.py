import numpy as np
import argparse
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


def f1_m(y_true, y_pred):
    """
    Calculate macro F1 score for multi-class predictions.
    """
    y_pred_binary = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred_binary, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_binary, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_binary), 'float'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)


def build_lstm_model(input_shape, num_classes=3, dropout_rate=0.4, l2_rate=1e-4):
    """
    Build the stacked LSTM model with added regularization:
      - Two LSTM layers (128 and 64 units) with L2 regularization
      - Dropout between layers increased to dropout_rate
      - Dense output with Softmax
    """
    l2_reg = regularizers.l2(l2_rate)
    model = Sequential()
    model.add(
        LSTM(
            units=128,
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(
        LSTM(
            units=64,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(
        Dense(
            units=num_classes,
            activation='softmax',
            kernel_regularizer=l2_reg
        )
    )
    return model


def main(args):
    print("🚀 Starting LSTM training…")

    # Load training data
    X_train = np.load(args.x_train)
    y_train = np.load(args.y_train)

    # Compute class weights
    y_labels = np.argmax(y_train, axis=1)
    classes = np.unique(y_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_labels)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Using class weights: {class_weight_dict}")

    # Build and compile model with gradient clipping
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_lstm_model(
        input_shape,
        num_classes,
        dropout_rate=args.dropout_rate,
        l2_rate=args.l2_rate
    )
    optimizer = Adam(learning_rate=args.learning_rate, clipnorm=args.clipnorm)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', f1_m]
    )

    # Setup callbacks with adjusted regularization schedule
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=args.checkpoint,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=args.log_dir,
            histogram_freq=1
        )
    ]

    # Train
    model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save final model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)
    print(f"✅ Model saved to {args.output_model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the LSTM anomaly detection model')
    parser.add_argument('--x_train',        type=str,   default='data/processed/X_train.npy',      help='Path to X_train NumPy file')
    parser.add_argument('--y_train',        type=str,   default='data/processed/y_train.npy',      help='Path to y_train (one-hot) NumPy file')
    parser.add_argument('--checkpoint',     type=str,   default='checkpoints/best_lstm.h5', help='Path to save best model checkpoint')
    parser.add_argument('--output_model',   type=str,   default='models/final_lstm.h5', help='Path to save the final model')
    parser.add_argument('--log_dir',        type=str,   default='logs/fit',        help='TensorBoard log directory')
    parser.add_argument('--learning_rate',  type=float, default=1e-4,             help='Initial learning rate for Adam optimizer')
    parser.add_argument('--clipnorm',       type=float, default=1.0,              help='Gradient clipping norm')
    parser.add_argument('--dropout_rate',   type=float, default=0.4,              help='Dropout rate for LSTM layers')
    parser.add_argument('--l2_rate',        type=float, default=1e-4,             help='L2 regularization factor')
    parser.add_argument('--batch_size',     type=int,   default=32,               help='Batch size for training')
    parser.add_argument('--epochs',         type=int,   default=50,               help='Maximum number of epochs')
    parser.add_argument('--validation_split', type=float, default=0.2,            help='Fraction of data for validation')
    args = parser.parse_args()
    main(args)
