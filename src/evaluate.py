#!/usr/bin/env python3
"""
Evaluation pipeline for ORION models.

Produces per-model:
  - classification report (precision / recall / F1 per class)
  - confusion matrix (PNG)
  - PR curves per class (PNG)
  - metrics JSON

And a cross-model comparison bar chart + summary JSON.

Usage:
  python src/evaluate.py                       # evaluate all known models
  python src/evaluate.py --models lstm_bi      # evaluate one model
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

# ---- allow imports from project root -------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common import CUSTOM_OBJECTS  # noqa: E402
from src.config import MODEL_REGISTRY as _REG, RESULTS_DIR  # noqa: E402

import tensorflow as tf  # noqa: E402

# ---------- model registry (derived from src.config) ----------
MODELS = {
    name: {
        "model_path": cfg["model_path"],
        "X_train": cfg["X_train"],
        "y_train": cfg["y_train"],
        "X_test": cfg["X_test"],
        "y_test": cfg["y_test"],
        "labels": cfg["labels"],
    }
    for name, cfg in _REG.items()
}


def load_model(path):
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)


# ---------- clinical metric helpers ----------

def sensitivity_at_specificity(y_true_onehot, probs, labels, thresholds=(0.90, 0.95, 0.99)):
    """For each class (one-vs-rest), find max sensitivity where specificity >= threshold."""
    results = {}
    for c, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, c], probs[:, c])
        specificity = 1.0 - fpr
        cls_results = {}
        for spec_thresh in thresholds:
            mask = specificity >= spec_thresh
            best_sens = float(tpr[mask].max()) if mask.any() else 0.0
            key = f"sens@spec{int(spec_thresh * 100)}"
            cls_results[key] = round(best_sens, 4)
        results[lbl] = cls_results
    return results


def compute_ece(y_true, y_pred, probs, labels, n_bins=10):
    """Expected Calibration Error on predicted-class confidence, plus per-class ECE."""
    pred_conf = probs[np.arange(len(probs)), y_pred]
    correct = (y_true == y_pred).astype(float)

    # Overall ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (pred_conf > bin_edges[b]) & (pred_conf <= bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = pred_conf[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)

    # Per-class ECE (one-vs-rest)
    num_classes = len(labels)
    y_true_onehot = np.eye(num_classes)[y_true]
    per_class_ece = {}
    for c, lbl in enumerate(labels):
        cls_conf = probs[:, c]
        cls_true = y_true_onehot[:, c]
        cls_ece = 0.0
        for b in range(n_bins):
            mask = (cls_conf > bin_edges[b]) & (cls_conf <= bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            bin_acc = cls_true[mask].mean()
            bin_conf = cls_conf[mask].mean()
            cls_ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
        per_class_ece[lbl] = round(float(cls_ece), 4)

    return round(float(ece), 4), per_class_ece


def compute_detection_latency(cfg, y_true, y_pred):
    """Measure windows between first anomaly ground-truth and first true-positive detection.

    Only runs when a groups file exists alongside the training data.
    Returns None if groups file is missing (graceful skip).
    """
    x_train_path = cfg["X_train"]
    groups_filename = os.path.basename(x_train_path).replace("X_", "groups_", 1)
    groups_path = os.path.join(os.path.dirname(x_train_path), groups_filename)

    if not os.path.isfile(groups_path):
        return None

    # Re-derive the test split indices (same seed/params as split_data.py)
    X_full = np.load(x_train_path)
    y_full = np.load(cfg["y_train"])
    groups = np.load(groups_path)
    y_int = np.argmax(y_full, axis=1) if y_full.ndim == 2 else y_full

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X_full, y_int, groups))
    test_groups = groups[test_idx]

    # y_true and y_pred are already aligned to test_idx ordering
    unique_patients = np.unique(test_groups)
    latencies = []
    detected = 0
    anomaly_patients = 0

    for pid in unique_patients:
        pmask = test_groups == pid
        pt_true = y_true[pmask]
        pt_pred = y_pred[pmask]

        # Skip patients with no anomaly windows
        if (pt_true == 0).all():
            continue
        anomaly_patients += 1

        # First window where ground truth is anomalous
        first_anomaly_idx = np.argmax(pt_true > 0)
        # First true-positive from that point onward
        remaining_true = pt_true[first_anomaly_idx:]
        remaining_pred = pt_pred[first_anomaly_idx:]
        tp_mask = (remaining_true > 0) & (remaining_pred > 0)

        if tp_mask.any():
            detected += 1
            first_tp_offset = np.argmax(tp_mask)
            latencies.append(int(first_tp_offset))

    if anomaly_patients == 0:
        return None

    result = {
        "anomaly_patients": anomaly_patients,
        "detected_patients": detected,
        "detection_rate": round(detected / anomaly_patients, 4),
    }
    if latencies:
        result["mean_latency_windows"] = round(float(np.mean(latencies)), 2)
        result["median_latency_windows"] = round(float(np.median(latencies)), 1)
        result["max_latency_windows"] = int(np.max(latencies))
    return result


def evaluate_model(name, cfg, out_dir):
    """Evaluate a single model. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"{'='*60}")

    if not os.path.isfile(cfg["model_path"]):
        print(f"  SKIP — model not found: {cfg['model_path']}")
        return None
    if not os.path.isfile(cfg["X_test"]):
        print(f"  SKIP — test data not found: {cfg['X_test']}")
        return None

    model = load_model(cfg["model_path"])
    X_test = np.load(cfg["X_test"])
    y_test = np.load(cfg["y_test"])
    labels = cfg["labels"]
    num_classes = len(labels)

    probs = model.predict(X_test, batch_size=512, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1) if y_test.ndim == 2 else y_test

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    # ---- confusion matrix plot ----
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} — Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ---- precision-recall curves ----
    fig, ax = plt.subplots(figsize=(7, 5))
    y_true_onehot = y_test if y_test.ndim == 2 else np.eye(num_classes)[y_true]
    pr_aucs = {}
    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, c], probs[:, c])
        ap = average_precision_score(y_true_onehot[:, c], probs[:, c])
        pr_aucs[labels[c]] = float(ap)
        ax.plot(recall, precision, label=f"{labels[c]} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{name} — Precision-Recall Curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "pr_curves.png"), dpi=150)
    plt.close(fig)

    # ---- sensitivity at fixed specificity ----
    sens_spec = sensitivity_at_specificity(y_true_onehot, probs, labels)

    # ---- calibration (ECE + reliability diagram) ----
    ece, per_class_ece = compute_ece(y_true, y_pred, probs, labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    n_bins = 10
    pred_conf = probs[np.arange(len(probs)), y_pred]
    correct = (y_true == y_pred).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for b in range(n_bins):
        mask = (pred_conf > bin_edges[b]) & (pred_conf <= bin_edges[b + 1])
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        else:
            bin_accs.append(float(correct[mask].mean()))
            bin_confs.append(float(pred_conf[mask].mean()))
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.6, edgecolor="black", label="Model")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{name} — Reliability Diagram (ECE={ece:.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "reliability.png"), dpi=150)
    plt.close(fig)

    # ---- detection latency ----
    latency = compute_detection_latency(cfg, y_true, y_pred)

    # ---- metrics JSON ----
    metrics = {
        "model": name,
        "model_path": cfg["model_path"],
        "accuracy": float(acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "pr_auc": pr_aucs,
        "per_class": {
            lbl: {
                "precision": float(report[lbl]["precision"]),
                "recall": float(report[lbl]["recall"]),
                "f1": float(report[lbl]["f1-score"]),
                "support": int(report[lbl]["support"]),
            }
            for lbl in labels
        },
        "confusion_matrix": cm.tolist(),
        "sensitivity_at_specificity": sens_spec,
        "calibration": {"ece": ece, "per_class_ece": per_class_ece},
    }
    if latency is not None:
        metrics["detection_latency"] = latency
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Macro F1:   {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1:{report['weighted avg']['f1-score']:.4f}")
    print(f"  PR-AUC:     {pr_aucs}")
    print(f"  Sensitivity@Specificity:")
    for lbl, vals in sens_spec.items():
        print(f"    {lbl}: {vals}")
    print(f"  Calibration ECE: {ece:.4f}")
    if latency is not None:
        print(f"  Detection latency: rate={latency['detection_rate']:.2%}"
              f"  mean={latency.get('mean_latency_windows', 'N/A')} windows")
    else:
        print("  Detection latency: skipped (no groups file)")
    print(f"  Saved to:   {model_dir}/")
    return metrics


def cross_model_comparison(all_metrics, out_dir):
    """Bar chart + summary JSON comparing all models."""
    if not all_metrics:
        return
    names = [m["model"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]
    f1s = [m["macro_f1"] for m in all_metrics]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, accs, w, label="Accuracy")
    ax.bar(x + w / 2, f1s, w, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nComparison chart saved to {out_dir}/comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ORION models")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names to evaluate (default: all)")
    parser.add_argument("--out_dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_metrics = []
    for name, cfg in MODELS.items():
        if args.models and name not in args.models:
            continue
        m = evaluate_model(name, cfg, args.out_dir)
        if m is not None:
            all_metrics.append(m)

    cross_model_comparison(all_metrics, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
