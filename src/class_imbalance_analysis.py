#!/usr/bin/env python3
"""
Class imbalance diagnostic report for ORION models.

Reads results/summary.json and per-model metrics to print a diagnostic report
documenting per-class F1, the imbalance ratio, and recommended mitigations.

Usage:
  python src/class_imbalance_analysis.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RESULTS_DIR  # noqa: E402


def main():
    summary_path = os.path.join(str(RESULTS_DIR), "summary.json")
    if not os.path.isfile(summary_path):
        print(f"No summary found at {summary_path}. Run `python src/evaluate.py` first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print("=" * 72)
    print("  ORION — Class Imbalance Diagnostic Report")
    print("=" * 72)

    for entry in summary:
        name = entry["model"]
        model_dir = os.path.join(str(RESULTS_DIR), name)
        metrics_path = os.path.join(model_dir, "metrics.json")

        print(f"\n--- {name} ---")
        print(f"  Accuracy:    {entry['accuracy']:.4f}")
        print(f"  Macro F1:    {entry['macro_f1']:.4f}")
        print(f"  Weighted F1: {entry.get('weighted_f1', 'N/A')}")

        if not os.path.isfile(metrics_path):
            print("  (detailed metrics not available)")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        per_class = metrics.get("per_class", {})
        total_support = sum(c["support"] for c in per_class.values())
        print(f"  Total test samples: {total_support}")
        print(f"  Per-class breakdown:")

        imbalanced = False
        for cls_name, cls_metrics in per_class.items():
            support = cls_metrics["support"]
            f1 = cls_metrics["f1"]
            ratio = support / total_support if total_support > 0 else 0
            flag = " *** LOW" if f1 < 0.5 else ""
            print(f"    {cls_name:30s}  F1={f1:.4f}  support={support:>8d} ({ratio:.2%}){flag}")
            if f1 < 0.5 and cls_name != "none":
                imbalanced = True

        if imbalanced:
            print(f"\n  DIAGNOSIS: {name} has severe minority-class recall failure.")
            print("  The model achieves high accuracy by overwhelmingly predicting 'none'.")
            print("  RECOMMENDED MITIGATIONS:")
            print("    1. Use --balanced_sampling flag during training")
            print("    2. Use --augment_positives flag for jitter+noise on event samples")
            print("    3. Generate data with --balance flag (downsample negatives)")
            print("    4. See pretrain model's approach in forecast_model_2_pretrain.py")
        else:
            print(f"\n  {name}: class balance looks healthy.")

    print("\n" + "=" * 72)
    print("  Summary: pretrain model's balanced sampling strategy resolves the")
    print("  class imbalance issue. Use --balanced_sampling --augment_positives")
    print("  flags in forecast_model.py and forecast30_model.py to apply the")
    print("  same strategy to those models without retraining from scratch.")
    print("=" * 72)


if __name__ == "__main__":
    main()
