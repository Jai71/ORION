"""Tests for src/config.py — path constants and model registry."""

from pathlib import Path

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    MODEL_REGISTRY,
    resolve,
)


def test_project_root_exists():
    """PROJECT_ROOT should be a real directory containing app.py."""
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "app.py").is_file()


def test_directory_constants_are_paths():
    """All directory constants should be Path objects."""
    for d in [DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
        assert isinstance(d, Path)


def test_resolve_returns_string():
    """resolve() should return a string anchored to PROJECT_ROOT."""
    result = resolve("some/path.txt")
    assert isinstance(result, str)
    assert str(PROJECT_ROOT) in result
    assert result.endswith("some/path.txt")


def test_model_registry_keys():
    """MODEL_REGISTRY should have the four expected model entries."""
    expected = {"lstm_bi", "forecast", "forecast30", "pretrain"}
    assert set(MODEL_REGISTRY.keys()) == expected


def test_model_registry_structure():
    """Each registry entry should have required keys."""
    required = {"display_name", "model_path", "X_train", "y_train",
                "X_test", "y_test", "labels", "has_attention"}
    for name, cfg in MODEL_REGISTRY.items():
        assert required.issubset(set(cfg.keys())), f"Missing keys in {name}: {required - set(cfg.keys())}"
        assert isinstance(cfg["labels"], list)
        assert isinstance(cfg["has_attention"], bool)
        assert len(cfg["labels"]) >= 3


def test_model_paths_are_absolute():
    """All paths in the registry should be absolute (anchored to PROJECT_ROOT)."""
    for name, cfg in MODEL_REGISTRY.items():
        for key in ["model_path", "X_train", "y_train", "X_test", "y_test"]:
            assert str(PROJECT_ROOT) in cfg[key], f"{name}.{key} is not anchored to PROJECT_ROOT"
