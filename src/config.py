"""Centralized path constants and model registry for ORION."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"


def resolve(relative_path: str) -> str:
    """Resolve a relative path against PROJECT_ROOT, returning a string."""
    return str(PROJECT_ROOT / relative_path)


MODEL_REGISTRY = {
    "lstm_bi": {
        "display_name": "lstm_bi (3-class)",
        "model_path": str(MODELS_DIR / "final_lstm.h5"),
        "X_train": str(DATA_DIR / "X_train.npy"),
        "y_train": str(DATA_DIR / "y_train.npy"),
        "X_test": str(DATA_DIR / "X_lstm_bi_split_test.npy"),
        "y_test": str(DATA_DIR / "y_lstm_bi_split_test.npy"),
        "labels": ["none", "anaphylaxis", "malignant_hyperthermia"],
        "has_attention": False,
    },
    "forecast": {
        "display_name": "forecast (4-class)",
        "model_path": str(MODELS_DIR / "forecast_model.h5"),
        "X_train": str(DATA_DIR / "X_pred_train.npy"),
        "y_train": str(DATA_DIR / "y_pred_train.npy"),
        "X_test": str(DATA_DIR / "X_forecast_split_test.npy"),
        "y_test": str(DATA_DIR / "y_forecast_split_test.npy"),
        "labels": ["none", "anaphylaxis", "malignant_hyperthermia", "respiratory_depression"],
        "has_attention": True,
    },
    "forecast30": {
        "display_name": "forecast30 (3-class)",
        "model_path": str(MODELS_DIR / "forecast30_model.h5"),
        "X_train": str(DATA_DIR / "X_pred30_train.npy"),
        "y_train": str(DATA_DIR / "y_pred30_train.npy"),
        "X_test": str(DATA_DIR / "X_forecast30_split_test.npy"),
        "y_test": str(DATA_DIR / "y_forecast30_split_test.npy"),
        "labels": ["none", "anaphylaxis", "malignant_hyperthermia"],
        "has_attention": True,
    },
    "pretrain": {
        "display_name": "pretrained (4-class)",
        "model_path": str(MODELS_DIR / "pretrained_model.keras"),
        "X_train": str(DATA_DIR / "X_pretrain.npy"),
        "y_train": str(DATA_DIR / "y_pretrain.npy"),
        "X_test": str(DATA_DIR / "X_pretrain_split_test.npy"),
        "y_test": str(DATA_DIR / "y_pretrain_split_test.npy"),
        "labels": ["none", "anaphylaxis", "malignant_hyperthermia", "respiratory_depression"],
        "has_attention": True,
    },
}
