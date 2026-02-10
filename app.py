#!/usr/bin/env python3
"""
ORION — Streamlit UI for medical anomaly detection.

Tabs:
  1. Predict          – single-sample prediction with vital signs plot
  2. Compare Models   – run one sample through all models
  3. Attention Viz    – attention heatmap overlay for forecast models
  4. Live Simulator   – animated synthetic patient with real-time predictions
  5. Batch Analysis   – confusion matrix / F1 on N samples
  6. Training Results – loss & accuracy curves from TensorBoard logs
"""

import glob
import json
import os
import random
import sys
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import tensorflow.keras.backend as K

# ---- project imports -------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from src.common import AttentionLayer, CUSTOM_OBJECTS  # noqa: E402
from src.config import MODEL_REGISTRY as _REG, RESULTS_DIR, LOGS_DIR  # noqa: E402
from predict import default_labels  # noqa: E402

FEATURE_NAMES = ["Heart Rate", "Blood Pressure", "SpO2", "Respiratory Rate", "Temperature"]

# ---- model / data registry (derived from src.config) -----------------------
MODEL_REGISTRY = {
    v["display_name"]: {
        "path": v["model_path"],
        "X": v["X_train"],
        "y": v["y_train"],
        "labels": v["labels"],
        "has_attention": v["has_attention"],
    }
    for v in _REG.values()
}


# ---- helpers ----------------------------------------------------------------
@st.cache_resource
def load_model_cached(path):
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)


@st.cache_data
def load_npy(path):
    return np.load(path, mmap_mode="r")


def available_models():
    return {k: v for k, v in MODEL_REGISTRY.items() if os.path.isfile(v["path"])}


def predict_probs(model, window):
    """Run a single window (T, F) through model and return probability array."""
    batch = np.expand_dims(window, 0)
    return model.predict(batch, verbose=0)[0]


def get_attention_weights(model, window):
    """Extract attention weights from the AttentionLayer in the model."""
    attn_layer = None
    for layer in model.layers:
        if isinstance(layer, AttentionLayer):
            attn_layer = layer
            break
    if attn_layer is None:
        return None
    # Build a sub-model up to just before the squeeze (we need softmax output)
    attn_input = attn_layer.input
    # Recompute attention scores
    W = attn_layer.W
    b = attn_layer.b
    sub_model = tf.keras.Model(inputs=model.input, outputs=attn_input)
    hidden = sub_model.predict(np.expand_dims(window, 0), verbose=0)
    e = np.tanh(hidden @ W.numpy() + b.numpy())
    e = e.squeeze(-1)  # (1, T)
    exp_e = np.exp(e - e.max(axis=-1, keepdims=True))
    alpha = exp_e / exp_e.sum(axis=-1, keepdims=True)
    return alpha[0]  # (T,)


def generate_live_vitals(t, baseline, anomaly_type=None, onset=None):
    """Generate one second of vital signs for the live simulator."""
    hr_b, bp_b, spo2_b, rr_b, temp_b = baseline
    hr = hr_b + np.random.normal(0, 1)
    bp = bp_b + np.random.normal(0, 2)
    spo2 = spo2_b + np.random.normal(0, 0.5)
    rr = rr_b + np.random.normal(0, 0.5)
    temp = temp_b + np.random.normal(0, 0.2)

    if anomaly_type and onset is not None and t >= onset:
        dt = t - onset
        ratio = min(dt / 300.0, 1.0)
        if anomaly_type == "anaphylaxis":
            hr = hr_b * (1 + 0.30 * ratio) + np.random.normal(0, 1)
            bp = bp_b * (1 - 0.30 * ratio) + np.random.normal(0, 2)
            spo2 = spo2_b * (1 - 0.10 * ratio) + np.random.normal(0, 0.5)
            rr = rr_b * (1 + 0.30 * ratio) + np.random.normal(0, 0.5)
            temp = temp_b + 0.2 * ratio + np.random.normal(0, 0.2)
        elif anomaly_type == "malignant_hyperthermia":
            hr = hr_b * (1 + 0.40 * ratio) + np.random.normal(0, 1)
            bp = bp_b * (1 - 0.15 * ratio) + np.random.normal(0, 2)
            spo2 = spo2_b * (1 - 0.07 * ratio) + np.random.normal(0, 0.5)
            rr = rr_b * (1 + 0.30 * ratio) + np.random.normal(0, 0.5)
            temp = temp_b + 1.0 * ratio + np.random.normal(0, 0.2)
        elif anomaly_type == "respiratory_depression":
            hr = hr_b * (1 - 0.08 * ratio) + np.random.normal(0, 1)
            bp = bp_b * (1 - 0.08 * ratio) + np.random.normal(0, 2)
            spo2_r = max(0, (dt - 60) / 240.0) if dt > 60 else 0
            spo2 = spo2_b * (1 - 0.12 * min(spo2_r, 1.0)) + np.random.normal(0, 0.5)
            rr = rr_b + np.random.normal(0, 0.5)
            temp = temp_b + np.random.normal(0, 0.2)
    return [hr, bp, spo2, rr, temp]


# ---- page config ------------------------------------------------------------
st.set_page_config(page_title="ORION", page_icon="🏥", layout="wide")
st.title("ORION — Medical Anomaly Detection")

tabs = st.tabs([
    "Predict", "Compare Models", "Attention Viz",
    "Live Simulator", "Batch Analysis", "Training Results",
])

models = available_models()

# ========================= TAB 1: PREDICT ====================================
with tabs[0]:
    st.header("Single-Sample Prediction")
    if not models:
        st.warning("No trained models found in models/")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            model_name = st.selectbox("Model", list(models.keys()), key="pred_model")
            cfg = models[model_name]
            X_data = load_npy(cfg["X"])
            n_samples = X_data.shape[0]
            idx = st.number_input("Sample index", 0, n_samples - 1, 0, key="pred_idx")
            run_pred = st.button("Predict", key="pred_go")

        if run_pred:
            window = np.array(X_data[idx])
            model = load_model_cached(cfg["path"])
            probs = predict_probs(model, window)
            labels = cfg["labels"]
            pred_class = int(np.argmax(probs))

            with col2:
                st.subheader(f"Predicted: **{labels[pred_class]}** ({probs[pred_class]:.1%})")
                # probability bar chart
                fig = go.Figure(go.Bar(
                    x=[f"{l}" for l in labels],
                    y=probs.tolist(),
                    marker_color=["#ef4444" if i == pred_class else "#94a3b8" for i in range(len(labels))],
                ))
                fig.update_layout(yaxis_title="Probability", yaxis_range=[0, 1], height=300)
                st.plotly_chart(fig, use_container_width=True)

            # vital signs plot (only if 5 features)
            if window.shape[1] == 5:
                st.subheader("Input Vital Signs (20-second window)")
                fig2 = go.Figure()
                for f_idx, fname in enumerate(FEATURE_NAMES):
                    fig2.add_trace(go.Scatter(
                        x=list(range(window.shape[0])),
                        y=window[:, f_idx].tolist(),
                        mode="lines+markers", name=fname,
                    ))
                fig2.update_layout(xaxis_title="Timestep (s)", yaxis_title="Value", height=350)
                st.plotly_chart(fig2, use_container_width=True)
            elif window.shape[1] == 20:
                st.info("This model uses 20 augmented features — showing raw feature chart is not applicable.")


# ========================= TAB 2: COMPARE MODELS =============================
with tabs[1]:
    st.header("Cross-Model Comparison")
    if len(models) < 2:
        st.warning("Need at least 2 trained models for comparison.")
    else:
        # pick a dataset that most models share
        compare_idx = st.number_input("Sample index (from X_train.npy)", 0, 100000, 42, key="cmp_idx")
        if st.button("Compare", key="cmp_go"):
            results = []
            for mname, mcfg in models.items():
                X = load_npy(mcfg["X"])
                if compare_idx >= X.shape[0]:
                    results.append({"Model": mname, "Prediction": "N/A (index out of range)", "Confidence": ""})
                    continue
                window = np.array(X[compare_idx])
                m = load_model_cached(mcfg["path"])
                probs = predict_probs(m, window)
                pred = int(np.argmax(probs))
                results.append({
                    "Model": mname,
                    "Prediction": mcfg["labels"][pred],
                    "Confidence": f"{probs[pred]:.1%}",
                    "All Probs": {l: f"{p:.4f}" for l, p in zip(mcfg["labels"], probs.tolist())},
                })

            for r in results:
                all_p = r.pop("All Probs", {})
                st.markdown(f"**{r['Model']}** → {r['Prediction']} ({r['Confidence']})")
                if all_p:
                    st.json(all_p)


# ========================= TAB 3: ATTENTION VIZ ==============================
with tabs[2]:
    st.header("Attention Visualization")
    attn_models = {k: v for k, v in models.items() if v["has_attention"]}
    if not attn_models:
        st.warning("No attention-based models found.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            attn_model_name = st.selectbox("Model", list(attn_models.keys()), key="attn_model")
            acfg = attn_models[attn_model_name]
            X_attn = load_npy(acfg["X"])
            attn_idx = st.number_input("Sample index", 0, X_attn.shape[0] - 1, 0, key="attn_idx")
            run_attn = st.button("Visualize", key="attn_go")

        if run_attn:
            window = np.array(X_attn[attn_idx])
            model = load_model_cached(acfg["path"])
            probs = predict_probs(model, window)
            weights = get_attention_weights(model, window)
            labels = acfg["labels"]
            pred = int(np.argmax(probs))

            with col2:
                st.subheader(f"Predicted: **{labels[pred]}** ({probs[pred]:.1%})")

            if weights is not None:
                # attention heatmap
                fig = go.Figure(go.Heatmap(
                    z=[weights.tolist()],
                    x=[f"t={i}" for i in range(len(weights))],
                    y=["Attention"],
                    colorscale="Reds",
                ))
                fig.update_layout(title="Attention Weights Over 20 Timesteps", height=200)
                st.plotly_chart(fig, use_container_width=True)

                # overlay on vital signs (if 5 features)
                if window.shape[1] == 5:
                    fig2 = go.Figure()
                    # normalize weights for visual opacity
                    w_norm = weights / weights.max()
                    for f_idx, fname in enumerate(FEATURE_NAMES):
                        fig2.add_trace(go.Scatter(
                            x=list(range(20)), y=window[:, f_idx].tolist(),
                            mode="lines+markers", name=fname,
                            marker=dict(
                                size=8,
                                color=w_norm.tolist(),
                                colorscale="Reds", cmin=0, cmax=1,
                            ),
                        ))
                    fig2.update_layout(
                        title="Vital Signs with Attention Overlay",
                        xaxis_title="Timestep (s)", yaxis_title="Value", height=400,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("Could not extract attention weights.")


# ========================= TAB 4: LIVE SIMULATOR =============================
with tabs[3]:
    st.header("Live Patient Simulator")
    col1, col2 = st.columns([1, 3])
    with col1:
        sim_model_name = st.selectbox("Model", list(models.keys()), key="sim_model")
        anomaly_choice = st.selectbox("Anomaly to inject", ["none", "anaphylaxis", "malignant_hyperthermia", "respiratory_depression"], key="sim_anomaly")
        onset_time = st.slider("Onset time (s)", 20, 120, 40, key="sim_onset")
        sim_duration = st.slider("Duration (s)", 40, 300, 120, key="sim_dur")
        start_sim = st.button("Start Simulation", key="sim_go")

    if start_sim:
        scfg = models[sim_model_name]
        sim_model = load_model_cached(scfg["path"])
        labels = scfg["labels"]
        n_features = sim_model.input_shape[-1]
        window_len = sim_model.input_shape[-2]  # typically 20

        baseline = [
            random.uniform(65, 85),   # HR
            random.uniform(110, 130), # BP
            random.uniform(96, 100),  # SpO2
            random.uniform(12, 16),   # RR
            random.uniform(36.5, 37.5),  # Temp
        ]

        anom = anomaly_choice if anomaly_choice != "none" else None
        onset = onset_time if anom else None

        chart_placeholder = col2.empty()
        pred_placeholder = col2.empty()

        history = []
        for t in range(sim_duration):
            vitals = generate_live_vitals(t, baseline, anom, onset)
            # If model expects 20 features (augmented), pad with zeros for demo
            if n_features == 20:
                row = vitals + [0.0] * 15
            else:
                row = vitals
            history.append(row)

            if len(history) >= window_len:
                window = np.array(history[-window_len:], dtype=np.float32)
                probs = predict_probs(sim_model, window)
                pred = int(np.argmax(probs))
                color = "🟢" if pred == 0 else "🔴"
                pred_placeholder.markdown(
                    f"**t={t}s** — {color} **{labels[pred]}** ({probs[pred]:.1%})"
                )

            # Plot last 60 seconds of vitals
            plot_history = history[-60:]
            fig = go.Figure()
            for f_idx, fname in enumerate(FEATURE_NAMES):
                fig.add_trace(go.Scatter(
                    x=list(range(max(0, t - len(plot_history) + 1), t + 1)),
                    y=[row[f_idx] for row in plot_history],
                    mode="lines", name=fname,
                ))
            if anom and onset and t >= onset:
                fig.add_vline(x=onset, line_dash="dash", line_color="red",
                              annotation_text="Anomaly onset")
            fig.update_layout(
                xaxis_title="Time (s)", yaxis_title="Value",
                height=350, margin=dict(t=30, b=30),
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.15)


# ========================= TAB 5: BATCH ANALYSIS =============================
with tabs[4]:
    st.header("Batch Analysis")
    if not models:
        st.warning("No trained models found.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            batch_model = st.selectbox("Model", list(models.keys()), key="batch_model")
            bcfg = models[batch_model]
            X_batch = load_npy(bcfg["X"])
            max_n = min(X_batch.shape[0], 50000)
            n_samples = st.slider("Number of samples", 100, max_n, min(5000, max_n), key="batch_n")
            show_misclass = st.checkbox("Show only misclassified", key="batch_mis")
            run_batch = st.button("Run Batch", key="batch_go")

        if run_batch:
            model = load_model_cached(bcfg["path"])
            y_data = load_npy(bcfg["y"])
            X_sub = np.array(X_batch[:n_samples])
            y_sub = np.array(y_data[:n_samples])
            labels = bcfg["labels"]

            probs = model.predict(X_sub, batch_size=512, verbose=0)
            y_pred = np.argmax(probs, axis=1)
            y_true = np.argmax(y_sub, axis=1) if y_sub.ndim == 2 else y_sub

            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)

            with col2:
                st.metric("Accuracy", f"{acc:.4f}")
                st.text(report)

            # confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            fig = go.Figure(go.Heatmap(
                z=cm.tolist(), x=labels, y=labels,
                colorscale="Blues", text=cm.tolist(), texttemplate="%{text}",
            ))
            fig.update_layout(
                title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            if show_misclass:
                mis = np.where(y_pred != y_true)[0]
                st.subheader(f"Misclassified samples ({len(mis)})")
                for i in mis[:20]:
                    st.write(f"  idx={i}: true={labels[y_true[i]]}, pred={labels[y_pred[i]]} (conf={probs[i][y_pred[i]]:.3f})")
                if len(mis) > 20:
                    st.write(f"  ... and {len(mis) - 20} more")


# ========================= TAB 6: TRAINING RESULTS ===========================
with tabs[5]:
    st.header("Training Results")

    # Check for evaluation results JSON
    results_dir = str(RESULTS_DIR)
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        st.subheader("Evaluation Summary")
        for m in summary:
            st.markdown(f"**{m['model']}** — Accuracy: {m['accuracy']:.4f}, Macro F1: {m['macro_f1']:.4f}")
        # comparison chart
        comp_img = os.path.join(results_dir, "comparison.png")
        if os.path.isfile(comp_img):
            st.image(comp_img, caption="Cross-Model Comparison")

        # per-model details
        for m in summary:
            model_dir = os.path.join(results_dir, m["model"])
            with st.expander(f"{m['model']} — Details"):
                metrics_path = os.path.join(model_dir, "metrics.json")
                if os.path.isfile(metrics_path):
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    st.json(metrics)
                cm_img = os.path.join(model_dir, "confusion_matrix.png")
                if os.path.isfile(cm_img):
                    st.image(cm_img, caption="Confusion Matrix")
                pr_img = os.path.join(model_dir, "pr_curves.png")
                if os.path.isfile(pr_img):
                    st.image(pr_img, caption="Precision-Recall Curves")
    else:
        st.info("Run `python src/evaluate.py` first to generate evaluation results.")

    # TensorBoard logs
    st.subheader("TensorBoard Logs")
    log_dirs = glob.glob(os.path.join(str(LOGS_DIR), "*/train/events.out.tfevents.*"))
    if log_dirs:
        st.write(f"Found {len(log_dirs)} training log(s). Launch TensorBoard:")
        st.code("tensorboard --logdir logs/")
    else:
        st.info("No TensorBoard logs found in logs/. Train a model first.")
