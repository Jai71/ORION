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

- **4 Model Architectures:** LSTM baseline, Bi-LSTM+Attention (3-class & 4-class), Pretrain+Fine-tune
- **Consolidated Data Generation:** Single script with presets (`--preset lstm_bi|forecast|forecast30|pretrain`)
- **Balanced Sampling:** Default-on class-balanced training for minority-class recall
- **Focal Loss + Class Weighting:** Per-class alpha from class frequencies
- **Input Validation:** NaN/Inf checks at inference boundaries (`validate_array()`)
- **Scaler Persistence:** Scalers saved alongside models for correct inference normalization
- **Clinical Evaluation Metrics:** Sensitivity@specificity, expected calibration error (ECE), detection latency
- **Reproducible Training:** NumPy, random, and TensorFlow seeds set in all training scripts
- **Test Suite:** 27 tests covering common layers, config, model architectures, and data pipeline
- **TensorBoard Integration:** Real-time training monitoring with histogram logging
- **Interpretable Attention:** See which timesteps influenced predictions
- **Streamlit UI:** 6-tab interactive dashboard for prediction, comparison, and visualization

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

### Generate Data + Train + Evaluate + UI

```bash
# 1. Generate synthetic data
python src/data_generation/generate.py --preset lstm_bi

# 2. Train a model (balanced sampling on by default)
python src/training/lstm_model_bi.py

# 3. Split data & evaluate
python src/split_data.py
python src/evaluate.py

# 4. Run tests
python -m pytest tests/ -v

# 5. Launch the UI
streamlit run app.py
```

---

## Project Structure

```
ORION/
├── app.py                              # Streamlit UI (6 tabs)
├── predict.py                          # CLI inference (imports from src.common)
├── requirements.txt                    # Pinned dependencies
├── pytest.ini                          # Pytest configuration
├── scripts/
│   └── verify_tf_gpu.py                # GPU verification script
├── src/
│   ├── __init__.py                     # Package init
│   ├── config.py                       # Centralized path constants, MODEL_REGISTRY
│   ├── common.py                       # Shared AttentionLayer, focal_loss, f1_m, CUSTOM_OBJECTS
│   ├── class_imbalance_analysis.py     # Class imbalance diagnostic report
│   ├── split_data.py                   # Stratified 80/20 train/test split
│   ├── evaluate.py                     # Evaluation pipeline (metrics + plots)
│   ├── data_generation/
│   │   └── generate.py                 # Consolidated data generation with presets
│   └── training/
│       ├── lstm_model_bi.py            # Bi-LSTM baseline (3-class)
│       ├── forecast_model.py           # Bi-LSTM+Attention (4-class, 60s horizon)
│       ├── forecast30_model.py         # Bi-LSTM+Attention (3-class, 30s horizon)
│       └── forecast_model_2_pretrain.py # Pretrain+fine-tune (4-class balanced)
├── tests/
│   ├── test_common.py                  # Tests for AttentionLayer, focal_loss, f1_m
│   ├── test_config.py                  # Tests for path constants, MODEL_REGISTRY
│   ├── test_models.py                  # Tests for model architecture builders
│   └── test_data_pipeline.py           # Tests for data generation, windowing, augmentation
├── data/processed/                     # Training arrays (.npy, generated)
├── models/                             # Saved trained models (.h5/.keras)
├── results/                            # Evaluation outputs (JSON + PNG)
├── checkpoints/                        # Best weights during training
└── logs/                               # TensorBoard logs
```

---

## Usage

### Generate Synthetic Data

```bash
python src/data_generation/generate.py --preset lstm_bi       # 3-class baseline (500 patients)
python src/data_generation/generate.py --preset forecast      # 4-class, 60s prediction horizon
python src/data_generation/generate.py --preset forecast30    # 3-class, 30s prediction horizon
python src/data_generation/generate.py --preset pretrain      # Balanced 4-class
```

### Train Models

```bash
# Bi-LSTM baseline (3-class)
python src/training/lstm_model_bi.py

# Forecast model (4-class, balanced sampling on by default)
python src/training/forecast_model.py
python src/training/forecast_model.py --augment_positives     # with additional augmentation

# Forecast30 model (3-class, balanced sampling on by default)
python src/training/forecast30_model.py
python src/training/forecast30_model.py --augment_positives   # with additional augmentation

# Pretrain+fine-tune (4-class balanced)
python src/training/forecast_model_2_pretrain.py --pretrain   # pretrain phase
python src/training/forecast_model_2_pretrain.py              # fine-tune phase
```

### Run Inference (Python)

```python
from tensorflow.keras.models import load_model
from src.common import CUSTOM_OBJECTS
import numpy as np

model = load_model('models/forecast_model.h5', custom_objects=CUSTOM_OBJECTS)

X_test = np.load('data/processed/X_pred_test.npy')  # Shape: (N, 20, 5)
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)

class_names = ['none', 'anaphylaxis', 'malignant_hyperthermia', 'respiratory_depression']
for i, (pred, prob) in enumerate(zip(predicted_classes, predictions)):
    print(f"Window {i}: {class_names[pred]} ({prob[pred]:.2%})")
```

### Run Inference (CLI)

```bash
python predict.py --model models/final_lstm.h5 --input data/processed/X_train.npy --sample-index 0
python predict.py --model models/forecast_model.h5 --input data/processed/X_pred_train.npy --sample-index 12 --top-k 4
```

---

## Configuration

All training scripts accept command-line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Maximum training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | `1e-4` | Adam learning rate |
| `--dropout_rate` | 0.4 | Dropout probability |
| `--clipnorm` | 1.0 | Gradient clipping norm |
| `--l2_rate` | `1e-4` | L2 regularization weight |
| `--focal_gamma` | 2.0 | Focal loss focusing parameter |
| `--focal_alpha` | per-class | Focal loss alpha (scalar override) |
| `--validation_split` | 0.2 | Fraction of data for validation |
| `--balanced_sampling` | on | Balance classes per batch (forecast/forecast30) |
| `--augment_positives` | off | Augment minority-class samples (opt-in) |

Use `--no-balanced_sampling` to disable balanced sampling on forecast/forecast30 models.

---

## Model Architectures

**1. Bi-LSTM Baseline** (`lstm_model_bi.py` — 3-class)
```
Input(20,5) -> Bi-LSTM(128) -> Dropout -> Bi-LSTM(64) -> Dropout -> Dense(3, softmax)
```

**2. Bi-LSTM + Attention** (`forecast_model.py` — 4-class)
```
Input(20,5) -> Bi-LSTM(128) -> Dropout -> Bi-LSTM(64) -> Dropout -> Attention -> Dense(4, softmax)
```

**3. Forecast30** (`forecast30_model.py` — 3-class, 30s horizon)
```
Same Bi-LSTM+Attention architecture as #2, 3-class output
```

**4. Pretrain+Fine-tune** (`forecast_model_2_pretrain.py` — 4-class)
```
Same Bi-LSTM+Attention architecture, with balanced per-batch sampling via tf.data interleave
+ on-the-fly augmentation (jitter ±3 timesteps, Gaussian noise sigma=0.02 on positives)
```

All attention-based models use a custom `AttentionLayer` (tanh scoring -> softmax -> weighted sum over time axis) defined in `src/common.py`.

---

## Evaluation

Run the evaluation pipeline to generate metrics and plots:

```bash
python src/split_data.py      # 80/20 stratified split (seed=42)
python src/evaluate.py         # metrics JSON + confusion matrices + PR curves
```

Results are saved to `results/` with per-model subdirectories and a cross-model `comparison.png`.

Clinical metrics include sensitivity at fixed specificity thresholds, expected calibration error (ECE), and detection latency analysis.

---

## Expected Performance

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Bi-LSTM (3-class) | ~93% | ~0.93 | Solid baseline, balanced per-class F1 |
| Pretrained (4-class) | ~99% | ~0.99 | Best overall, balanced sampling + augmentation |
| Forecast (4-class) | ~92% | ~0.93* | With `--balanced_sampling` (default) |
| Forecast30 (3-class) | ~96% | ~0.95* | With `--balanced_sampling` (default) |

*Without balanced sampling, forecast/forecast30 macro F1 drops to ~0.24/~0.33 due to class-0 dominance. Balanced sampling is now **on by default** to prevent this.

**Note:** Synthetic data only — real-world performance will differ.

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

## Testing

```bash
python -m pytest tests/ -v
```

27 tests covering:
- `test_common.py` — AttentionLayer, focal_loss, f1_m (argmax+one_hot fix), validate_array
- `test_config.py` — Path constants, MODEL_REGISTRY completeness
- `test_models.py` — Model architecture builders (output shapes, layer counts)
- `test_data_pipeline.py` — Data generation, windowing, augmentation (vectorized)

---

## Troubleshooting

**Model predicts only class 0:**
-> Balanced sampling is now on by default for forecast/forecast30. If you disabled it with `--no-balanced_sampling`, re-enable it. The pretrain model uses its own balanced `tf.data` pipeline.

**Shape mismatch errors:**
-> Ensure input is `(N, 20, 5)` for base models or `(N, 20, 20)` for augmented-feature models.

**NaN/Inf in predictions:**
-> `validate_array()` checks inputs at inference boundaries. Ensure your input data has no NaN/Inf values. Check scaler files are present alongside the model.

**Low accuracy:**
-> Try the pretrain variant or increase epochs. Ensure data was generated with the correct preset for the model being trained.

---

**Last Updated:** 2026-02-12
