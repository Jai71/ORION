# ORION — Medical Anomaly Detection System

**AI-powered early warning system for detecting adverse medical events from patient vital signs**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![Metal GPU](https://img.shields.io/badge/GPU-Apple%20Metal-silver.svg)](https://developer.apple.com/metal/)

---

## Overview

ORION is a deep learning system that predicts life-threatening medical emergencies from continuous patient monitoring data:

- **Anaphylaxis** (severe allergic reaction)
- **Malignant Hyperthermia** (dangerous fever crisis)
- **Respiratory Depression** (breathing failure)

**How it works:**
1. Monitors 5 vital signs: Heart Rate, Blood Pressure, Oxygen Saturation, Respiratory Rate, Temperature
2. Analyzes 20-second windows using LSTM neural networks
3. Predicts anomaly type with probability scores
4. Provides early warning before critical deterioration

---

## Features

- **3 Model Architectures:** LSTM baseline, Bi-LSTM+Attention, Bi-LSTM+Augmentation
- **Synthetic Data Generation:** Realistic patient simulations
- **Imbalance Handling:** Focal loss + class weighting
- **TensorBoard Integration:** Real-time training monitoring
- **Interpretable Attention:** See which timesteps influenced predictions

---

## Quick Start

### Installation

```bash
cd ORION
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify GPU (Apple Silicon)
python scripts/verify_tf_gpu.py
```

### Train + Evaluate + UI

```bash
# 1. Train a model
python src/training/lstm_model_bi.py --epochs 50 --batch_size 128

# 2. Split data & evaluate
python src/split_data.py
python src/evaluate.py

# 3. Launch the UI
streamlit run app.py
```

---

## Project Structure

```
ORION/
├── app.py                   # Streamlit UI (6 tabs)
├── predict.py               # Inference helper script
├── requirements.txt         # Pinned dependencies
├── scripts/
│   └── verify_tf_gpu.py     # GPU verification script
├── src/
│   ├── split_data.py        # Deterministic train/test split
│   ├── evaluate.py          # Evaluation pipeline (metrics + plots)
│   ├── data_generation/     # Synthetic data generation scripts
│   └── training/            # Model training scripts
│       ├── lstm_model_bi.py           # Bi-LSTM (3-class)
│       ├── forecast_model_2_pretrain.py # Bi-LSTM+Attention (4-class, pretrain+fine-tune)
│       ├── forecast_model_3.py        # Hard-negative mining variant
│       └── forecast30_model.py        # 30s forecast variant
├── data/processed/          # Training arrays (.npy)
├── models/                  # Saved trained models
├── results/                 # Evaluation outputs (JSON + PNG)
├── checkpoints/             # Best weights during training
└── logs/                    # TensorBoard logs
```

---

## Usage

### Train Baseline Model

```bash
python src/training/lstm_model.py \
  --x_train data/processed/X_train.npy \
  --y_train data/processed/y_train.npy \
  --epochs 50 \
  --batch_size 32
```

### Train Advanced Model (Recommended)

```bash
# Generate 4-class data
python src/data_generation/data_generation_pred60.py

# Train with attention + focal loss
python src/training/forecast_model.py \
  --x_train data/processed/X_pred_train.npy \
  --y_train data/processed/y_pred_train.npy \
  --epochs 100
```

### Run Inference

```python
from tensorflow.keras.models import load_model
from src.training.forecast_model import AttentionLayer, focal_loss, f1_m
import numpy as np

model = load_model('models/forecast_model.h5', custom_objects={
    'AttentionLayer': AttentionLayer,
    'loss': focal_loss(),
    'f1_m': f1_m
})

X_test = np.load('my_test_windows.npy')  # Shape: (N, 20, 5)
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

class_names = ['none', 'anaphylaxis', 'malignant_hyperthermia', 'respiratory_depression']
for i, (pred, prob) in enumerate(zip(predicted_classes, predictions)):
    print(f"Window {i}: {class_names[pred]} ({prob[pred]:.2%})")
```

---

## Configuration

All models use command-line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50-100 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | `1e-4` | Adam learning rate |
| `--dropout_rate` | 0.4 | Dropout probability |
| `--focal_gamma` | 2.0 | Focal loss parameter (forecast models) |

---

## Model Architectures

**1. Baseline LSTM** (118K params, 3-class)
```
Input -> LSTM(128) -> Dropout -> LSTM(64) -> Dropout -> Dense(3)
```

**2. Bi-LSTM + Attention** (305K params, 4-class)
```
Input -> Bi-LSTM(128) -> Dropout -> Bi-LSTM(64) -> Dropout -> Attention -> Dense(4)
```

**3. With Augmentation** (same as #2 + time-series augmentation)

---

## Troubleshooting

**Model predicts only class 0:**
-> Use `forecast_model.py` with focal loss

**Shape mismatch errors:**
-> Ensure input is `(N, 20, 5)`

**Low accuracy:**
-> Try augmentation variant or increase epochs

---

## Evaluation

Run the evaluation pipeline to generate metrics and plots:

```bash
python src/split_data.py      # 80/20 stratified split (seed=42)
python src/evaluate.py         # metrics JSON + confusion matrices + PR curves
```

Results are saved to `results/` with per-model subdirectories and a cross-model `comparison.png`.

---

## Streamlit UI

```bash
streamlit run app.py
```

Six tabs:
1. **Predict** — single-sample prediction with vital signs plot
2. **Compare Models** — run one sample through all models side-by-side
3. **Attention Viz** — heatmap showing which timesteps the model focused on
4. **Live Simulator** — animated synthetic patient with real-time predictions
5. **Batch Analysis** — confusion matrix and classification report on N samples
6. **Training Results** — evaluation metrics, plots, and TensorBoard log info

---

## Expected Performance

| Model | Accuracy | Macro F1 | Training Time (M4 Pro) |
|-------|----------|----------|------------------------|
| Bi-LSTM (3-class) | ~93% | ~0.93 | ~33 min/epoch |
| Pretrained (4-class balanced) | ~99% | ~0.99 | ~1 min/epoch |
| Forecast (4-class imbalanced) | ~92% | ~0.24* | ~25 min/epoch |
| Forecast30 (3-class) | ~96% | ~0.33* | ~25 min/epoch |

*Low macro F1 on imbalanced sets reflects class-0 dominance (99%+). Per-class PR-AUC and the pretrained model show the architecture works.

**Note:** Synthetic data only — real-world performance will differ.

---

**Last Updated:** 2026-02-10
