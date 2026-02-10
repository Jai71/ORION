# CLAUDE.md — ORION

Medical anomaly detection system for surgical complications.
Detects anaphylaxis, malignant hyperthermia, and respiratory depression from synthetic patient vital signs using Bi-LSTM models on Apple Silicon GPU.

---

## Repository Map

```
ORION/
├── app.py                              # Streamlit UI (6 tabs)
├── predict.py                          # CLI inference (imports from src.common)
├── requirements.txt                    # Pinned Python dependencies
├── pytest.ini                          # Pytest configuration
├── README.md
├── .claudeignore                       # Excludes heavy data/model dirs from indexing
├── .gitignore                          # Excludes data, models, results from git
├── archive/                            # Archived text snapshots of earlier code variants
├── results/                            # Evaluation outputs
│   ├── summary.json                    # Cross-model accuracy & F1 scores
│   ├── comparison.png                  # Bar chart comparing all models
│   ├── lstm_bi/                        # Per-model metrics, confusion matrix, PR curves
│   ├── forecast/
│   ├── forecast30/
│   └── pretrain/
├── scripts/
│   └── verify_tf_gpu.py                # TF Metal GPU verification + matmul benchmark
├── src/
│   ├── __init__.py                     # Package init
│   ├── config.py                       # Centralized path constants, MODEL_REGISTRY
│   ├── common.py                       # Shared AttentionLayer, focal_loss, f1_m, CUSTOM_OBJECTS
│   ├── class_imbalance_analysis.py     # Diagnostic report for class imbalance
│   ├── evaluate.py                     # Evaluation pipeline (imports from src.config, src.common)
│   ├── split_data.py                   # Stratified 80/20 train/test split
│   ├── data_generation/
│   │   ├── generate.py                 # CONSOLIDATED: parameterized data generation with presets
│   │   ├── data_generation.py          # (deprecated) Core: 500 patients, 3-class
│   │   ├── data_generation_pred30.py   # (deprecated) 30s prediction horizon
│   │   ├── data_generation_pred60.py   # (deprecated) 60s prediction horizon
│   │   ├── data_generation_2.py        # (deprecated) Iteration variant with jitter
│   │   ├── data_generation_pred60_2.py # (deprecated) 60s + augment + hard negatives + balance
│   │   ├── data_generation_pred60_3.py # (deprecated) 60s + augment + jitter
│   │   ├── balanced_dataset_generation.py # (deprecated) Full class balancing
│   │   └── unchanged_data_generation.py   # (deprecated) Raw CSV export
│   └── training/
│       ├── lstm_model_bi.py            # Primary: Bi-LSTM baseline (3-class)
│       ├── forecast_model.py           # Bi-LSTM+Attention (4-class, 60s horizon)
│       ├── forecast30_model.py         # Bi-LSTM+Attention (3-class, 30s horizon)
│       ├── forecast_model_2_pretrain.py # Pretrain+fine-tune with balanced sampling
│       ├── forecast_model_2.py         # Iteration variant
│       ├── forecast_model_3.py         # Iteration variant
│       ├── lstm_model.py              # Earlier unidirectional LSTM
│       └── lstm_model_2.py            # Iteration variant
├── tests/
│   ├── __init__.py
│   ├── test_common.py                  # Tests for AttentionLayer, focal_loss, f1_m
│   ├── test_config.py                  # Tests for path constants, MODEL_REGISTRY
│   ├── test_models.py                  # Tests for model architecture builders
│   └── test_data_pipeline.py           # Tests for data generation, windowing, augmentation
├── data/                               # ~15 GB .npy arrays (excluded from context)
├── models/                             # ~18 MB saved .h5/.keras weights
├── checkpoints/                        # ~14 MB training checkpoints
└── logs/                               # ~3.3 MB TensorBoard event files
```

---

## Architecture

```
Synthetic patient data                .npy arrays              Trained models           Consumers
─────────────────────                ────────────             ──────────────           ─────────
generate.py (presets)    ──────►  data/processed/  ──────►  models/*.h5/.keras  ──►  app.py (Streamlit)
(500 patients x 3000s)           X_*.npy, y_*.npy           (4 model variants)       predict.py (CLI)
                                       │                                              evaluate.py (metrics)
                                       ▼
                                 split_data.py
                                 (80/20 stratified)
                                       │
                                       ▼
                                 *_split_train.npy
                                 *_split_test.npy
```

All models consume windowed input of shape `(N, 20, F)` where `F` is 5 (raw vitals) or 20 (augmented features). Output is softmax probability over 3 or 4 classes.

---

## Shared Modules

### `src/config.py` — Centralized path constants
- `PROJECT_ROOT`: anchors all paths to the repository root
- `DATA_DIR`, `MODELS_DIR`, `CHECKPOINTS_DIR`, `LOGS_DIR`, `RESULTS_DIR`: directory constants
- `resolve(relative_path)`: resolves a relative path against `PROJECT_ROOT`
- `MODEL_REGISTRY`: single source of truth for all model configs (paths, labels, data paths)

### `src/common.py` — Shared custom Keras objects
- `AttentionLayer`: tanh scoring → softmax → weighted sum over time axis
- `focal_loss(gamma, alpha)`: focal loss for multi-class classification
- `f1_m(y_true, y_pred)`: macro F1 metric (type-safe variant with explicit cast)
- `CUSTOM_OBJECTS`: dict mapping string names to classes/functions (for model loading)

---

## Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| TensorFlow | 2.18.1 |
| TensorFlow Metal | 1.2.0 (Apple Silicon GPU) |
| NumPy | 2.0.2 |
| Pandas | 3.0.0 |
| scikit-learn | 1.8.0 |
| pytest | latest |
| Streamlit | latest (unpinned) |
| Plotly | latest (unpinned) |
| Matplotlib | latest (unpinned) |

---

## Entrypoints

| Command | Purpose |
|---------|---------|
| `streamlit run app.py` | Launch Streamlit UI (6 tabs) |
| `python predict.py --model <path> --input <npy> --sample-index N` | CLI inference |
| `python -m pytest tests/ -v` | Run test suite |
| `python src/data_generation/generate.py --preset lstm_bi` | Generate 3-class baseline data |
| `python src/data_generation/generate.py --preset forecast` | Generate 4-class 60s-horizon data |
| `python src/data_generation/generate.py --preset forecast30` | Generate 3-class 30s-horizon data |
| `python src/data_generation/generate.py --preset pretrain` | Generate balanced 4-class data |
| `python src/training/lstm_model_bi.py` | Train baseline Bi-LSTM (3-class) |
| `python src/training/forecast_model.py` | Train forecast model (4-class) |
| `python src/training/forecast_model.py --balanced_sampling` | Train with balanced sampling |
| `python src/training/forecast30_model.py` | Train forecast30 model (3-class) |
| `python src/training/forecast_model_2_pretrain.py [--pretrain]` | Train pretrain model |
| `python src/evaluate.py [--models lstm_bi forecast ...]` | Run evaluation pipeline |
| `python src/split_data.py` | Stratified 80/20 train/test split |
| `python src/class_imbalance_analysis.py` | Print class imbalance diagnostic report |
| `python scripts/verify_tf_gpu.py` | Verify TensorFlow sees Metal GPU |

---

## Model Definitions

### 1. lstm_bi — `src/training/lstm_model_bi.py`
- **Architecture**: Sequential — Bi-LSTM(128, return_seq) → Dropout → Bi-LSTM(64) → Dropout → Dense(3, softmax)
- **Classes**: 3 (none, anaphylaxis, malignant_hyperthermia)
- **Input shape**: (20, 5)
- **Output**: `models/final_lstm.h5`

### 2. forecast — `src/training/forecast_model.py`
- **Architecture**: Functional — Bi-LSTM(128, return_seq) → Dropout → Bi-LSTM(64, return_seq) → Dropout → AttentionLayer → Dense(4, softmax)
- **Classes**: 4 (adds respiratory_depression)
- **Input shape**: (20, 5) or (20, 20) with augmented features
- **Output**: `models/forecast_model.h5`
- **New flags**: `--balanced_sampling`, `--augment_positives`

### 3. forecast30 — `src/training/forecast30_model.py`
- **Architecture**: Same as forecast (Bi-LSTM+Attention)
- **Classes**: 3 (none, anaphylaxis, malignant_hyperthermia)
- **Input shape**: (20, 5)
- **Output**: `models/forecast30_model.h5`
- **New flags**: `--balanced_sampling`, `--augment_positives`

### 4. pretrain — `src/training/forecast_model_2_pretrain.py`
- **Architecture**: Same Bi-LSTM+Attention as forecast
- **Classes**: 4
- **Extra**: `--pretrain` flag toggles simple training mode vs balanced per-batch sampling via `tf.data` interleave + on-the-fly augmentation (jitter ±3 timesteps, Gaussian noise sigma=0.02 on positive samples)
- **Output**: `models/pretrained_model.keras`

All models use a custom `AttentionLayer` (tanh scoring → softmax → weighted sum over time axis), defined in `src/common.py`.

---

## Hyperparameters

### Training CLI defaults (shared across all 4 training scripts)

| Parameter | Default | Flag |
|-----------|---------|------|
| Learning rate | 1e-4 | `--learning_rate` |
| Gradient clip norm | 1.0 | `--clipnorm` |
| Dropout rate | 0.4 | `--dropout_rate` |
| L2 regularization | 1e-4 | `--l2_rate` |
| Focal loss gamma | 2.0 | `--focal_gamma` |
| Focal loss alpha | 0.25 | `--focal_alpha` |
| Batch size | 32 | `--batch_size` |
| Max epochs | 50 | `--epochs` |
| Validation split | 0.2 | `--validation_split` |

### Callbacks
- EarlyStopping: monitor=val_loss, patience=3, restore_best_weights=True
- ReduceLROnPlateau: factor=0.3, patience=2, min_lr=1e-6
- ModelCheckpoint: save_best_only=True
- TensorBoard: histogram_freq=1

### Data generation constants (hardcoded)

| Constant | Value | Location |
|----------|-------|----------|
| NUM_PATIENTS | 500 | All data_generation scripts |
| TIMESTEPS | 3000 (1 Hz, ~50 min) | All data_generation scripts |
| FEATURES | 5 (HR, BP, SpO2, RR, Temp) | All data_generation scripts |
| WINDOW | 20 seconds | All data_generation scripts |
| ANOMALY_PROBABILITY | 0.3 | All data_generation scripts |
| PREDICTION_HORIZON | 60s (pred60), 30s (pred30) | Horizon-specific scripts |

---

## Evaluation Results

From `results/summary.json`:

| Model | Accuracy | Macro F1 | Weighted F1 | Notes |
|-------|----------|----------|-------------|-------|
| lstm_bi | 0.9309 | 0.9314 | 0.9309 | Solid baseline, balanced per-class F1 |
| forecast | 0.9151 | **0.2417** | 0.9491 | Near-zero minority-class recall |
| forecast30 | 0.9580 | **0.3290** | 0.9761 | Predicts MH as zero across the board |
| pretrain | 0.9872 | 0.9871 | 0.9871 | Best overall, balanced sampling works |

The forecast and forecast30 models achieve high accuracy by overwhelmingly predicting "none" — their minority-class (anaphylaxis, MH, respiratory_depression) F1 scores are near zero.

---

## Risk Hotspots

1. **~~Duplicated custom layers~~** — RESOLVED. `AttentionLayer`, `focal_loss`, and `f1_m` now live in `src/common.py`. All training scripts, `predict.py`, `app.py`, and `evaluate.py` import from this single source.

2. **~~Near-identical data generation scripts~~** — RESOLVED. Consolidated into `src/data_generation/generate.py` with CLI presets (`--preset lstm_bi|forecast|forecast30|pretrain`). Old scripts retained with deprecation comments.

3. **~~No test suite~~** — RESOLVED. `tests/` directory with 4 test files covering common layers, config, model architectures, and data pipeline. Run with `python -m pytest tests/ -v`.

4. **~~Hardcoded paths~~** — RESOLVED. All paths now resolved via `src/config.py` (`PROJECT_ROOT`, `resolve()`, `MODEL_REGISTRY`). Scripts work from any working directory.

5. **~~Low macro-F1 on forecast/forecast30~~** — MITIGATED. `--balanced_sampling` and `--augment_positives` flags added to both `forecast_model.py` and `forecast30_model.py`. Retraining with these flags should match pretrain's ~0.99 macro F1.

6. **Large repo size** — See Repo Size Reduction Plan below.

---

## Repo Size Reduction Plan

The repository contains ~15 GB of generated `.npy` data files and ~32 MB of model weights/checkpoints. Recommendations:

1. **Move `data/` to cloud storage** (S3, GCS, or HuggingFace Datasets) with a download script (`scripts/download_data.py`). The consolidated `generate.py` can regenerate any dataset.
2. **Use Git LFS for `models/`** weights if they must stay in-repo. Currently gitignored.
3. **`results/` generated artifacts** are now in `.gitignore` (PNGs, per-model subdirectories).
4. **Keep `data/processed/.gitkeep`** so the directory structure exists after a fresh clone.
5. **All `.npy`, `.h5`, `.keras` files** are now covered by `.gitignore` catch-all rules.

---

## Ignored Paths

These directories are excluded from agent context (via `.claudeignore`) due to size or irrelevance:

| Path | Size | Contents |
|------|------|----------|
| `data/` | ~15 GB | `.npy` training/test arrays |
| `models/` | ~18 MB | Saved `.h5` / `.keras` model weights |
| `checkpoints/` | ~14 MB | Training checkpoint files |
| `logs/` | ~3.3 MB | TensorBoard event files |
| `.venv/` | — | Python virtual environment |
| `__pycache__/` | — | Bytecode cache |
| `.git/` | — | Git internals |
| `results/` | — | Evaluation PNGs and JSONs |
| `archive/` | — | Archived code snapshots |

---

## Commands

```bash
# Environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify GPU
python scripts/verify_tf_gpu.py

# Run tests
python -m pytest tests/ -v

# Generate synthetic data (consolidated script with presets)
python src/data_generation/generate.py --preset lstm_bi       # 3-class baseline
python src/data_generation/generate.py --preset forecast      # 4-class, 60s horizon
python src/data_generation/generate.py --preset forecast30    # 3-class, 30s horizon
python src/data_generation/generate.py --preset pretrain      # balanced 4-class

# Train models
python src/training/lstm_model_bi.py
python src/training/forecast_model.py
python src/training/forecast_model.py --balanced_sampling --augment_positives  # with class balance fix
python src/training/forecast30_model.py
python src/training/forecast30_model.py --balanced_sampling --augment_positives
python src/training/forecast_model_2_pretrain.py --pretrain   # pretrain phase
python src/training/forecast_model_2_pretrain.py              # fine-tune phase

# Split data for evaluation
python src/split_data.py

# Evaluate
python src/evaluate.py

# Class imbalance diagnostic
python src/class_imbalance_analysis.py

# Launch UI
streamlit run app.py

# CLI prediction
python predict.py --model models/final_lstm.h5 --input data/processed/X_train.npy --sample-index 0
python predict.py --model models/forecast_model.h5 --input data/processed/X_pred_train.npy --sample-index 12 --top-k 4
```
