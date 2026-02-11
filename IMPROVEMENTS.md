# ORION — Prioritized Improvement List

## 1. ROOT CAUSE of low macro-F1 on forecast/forecast30: Window-level label sparsity

This is the single most important issue. The problem is **not** just "class imbalance" — it's a ~160:1 negative-to-positive ratio created by the prediction-horizon windowing scheme.

**The math:**
- 500 patients, 30% anomaly rate = ~150 anomaly patients, ~350 normal
- Each patient produces ~2,920 windows (`3000 - 20 - 60 + 1`)
- **Normal patient**: all 2,920 windows labeled "none"
- **Anomaly patient**: only windows where `start + WINDOW <= onset < start + WINDOW + horizon` are positive. With horizon=60, that's roughly **60 positive windows out of 2,920** per anomaly patient
- Total positive: ~150 * 60 = **9,000** out of ~1,460,000 total = **0.6% positive rate**
- forecast30 is even worse: horizon=30 means ~30 positive windows per patient = **0.3% positive rate**

**Why lstm_bi works**: Per-patient labeling means ALL ~2,981 windows from an anomaly patient are positive → ~30% positive. Much more balanced.

**Why pretrain works**: It uses `balance=True` in data generation (downsamples negatives to 2x positives) AND uses 1:1 pos/neg interleaving in the `tf.data` pipeline.

**Why `class_weight` alone fails** (`forecast_model.py` line 147): With batch_size=32 and 0.6% positive rate, the expected number of positive samples per batch is **0.19**. Most batches contain zero positives. `class_weight` scales the loss but can't produce gradient signal from samples that aren't in the batch. The model converges to always-predict-"none" because that's all it sees.

**Fix**: The `--balanced_sampling` flag already exists. The real issue is that the **default** training path (without the flag) is broken by design for horizon-labeled data. Either:
- Make `--balanced_sampling` the default for forecast/forecast30
- Or apply `balance=True` at data generation time (as pretrain does) so the default training path receives reasonable data

---

## 2. `focal_loss` alpha doesn't address per-class imbalance

`focal_loss(gamma=2.0, alpha=0.25)` in `src/common.py`:

```python
weight = alpha * tf.math.pow(1 - y_pred, gamma)
```

Alpha is a **scalar** applied uniformly to all classes. In the original focal loss paper, alpha is a **per-class vector** (higher for rare classes). Here it's just a constant multiplier on the entire loss, providing zero class-rebalancing effect. The gamma term (down-weighting easy examples) helps somewhat but can't overcome a 160:1 ratio.

**Fix**: Make alpha a per-class vector derived from class frequencies, e.g. `alpha = [0.1, 0.3, 0.3, 0.3]` for 4-class, or compute it from `class_weight`.

---

## 3. No scaler persistence — inference on unscaled data silently produces wrong results

`generate.py` fits a `StandardScaler` and transforms the data, but the scaler is never saved. At inference time (`predict.py`, `app.py`), the model receives whatever's in the `.npy` file — which is pre-scaled from data generation. But:
- If someone generates new data and forgets to scale, predictions are garbage
- The live simulator (`app.py`) feeds raw unscaled vitals directly into the model
- The `predict.py` CLI has no scaling step

**Fix**: Save the scaler alongside the data (`joblib.dump`) and load it in `predict.py`/`app.py` before inference.

---

## 4. Live simulator feeds zeros for 20-feature models

`app.py`:
```python
if n_features == 20:
    row = vitals + [0.0] * 15  # padding with zeros!
```

The forecast model expects `[raw, deltas, rolling_mean, rolling_std]` — all 4 feature groups carry signal. Padding the last 15 features with zeros means the model receives data entirely unlike its training distribution. Predictions from the live simulator for the forecast model are meaningless.

**Fix**: Accumulate raw vitals in a buffer, then call `augment_features()` on the 20-step window before prediction.

---

## 5. Validation split is not shuffled — temporal leak + biased val set

`forecast_model.py` and `forecast30_model.py`:
```python
split_idx = int((1 - args.validation_split) * X.shape[0])
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]
```

Windows are generated patient-by-patient sequentially. The last 20% of windows all come from the last ~100 patients. This means:
- Train and val may have very different class distributions
- No patient-level separation — windows from the same patient can leak across sets

Even the Keras `validation_split=` parameter (used by `lstm_model_bi.py`) just takes the last 20% without shuffling.

**Fix**: Use `train_test_split` with `stratify` and patient-level grouping (e.g., `GroupShuffleSplit`) to prevent patient leakage.

---

## 6. EarlyStopping + ReduceLROnPlateau patience interaction

All training scripts use:
- `ReduceLROnPlateau(patience=2)`
- `EarlyStopping(patience=3)`

LR reduction fires after 2 stagnant epochs, but early stopping kills training after 3. The model gets at most **1 epoch** at the reduced learning rate before stopping. The LR schedule is effectively useless.

**Fix**: Either increase EarlyStopping patience to 6+ (so the model trains for several epochs at each reduced LR), or decrease ReduceLROnPlateau patience to 1.

---

## 7. No patient-level split in evaluation

`split_data.py` does a stratified window-level split. Windows from the same patient appear in both train and test sets. Since consecutive windows overlap by 19 out of 20 timesteps, the test set contains near-duplicates of training examples. This **inflates all reported metrics**.

**Fix**: Split at the patient level before windowing. Tag each window with its source patient ID and ensure all windows from one patient go to the same fold.

---

## 8. Evaluation doesn't measure what matters clinically

`evaluate.py` reports accuracy, macro F1, weighted F1, and PR curves. For a medical anomaly detector, the critical metrics are:
- **Sensitivity (recall) at a fixed specificity** — e.g., recall at 95% specificity
- **Detection latency** — how many seconds before onset does the model first predict positive?
- **Per-class recall** (especially for the rarest condition, respiratory_depression)
- **Calibration** — are the predicted probabilities reliable?

None of these are computed.

---

## 9. `f1_m` metric computes F1 on rounded one-hot probabilities per batch

`src/common.py`: `y_pred_ = K.round(y_pred)` rounds softmax outputs to 0/1 per class, then computes F1 per batch. This is a noisy, batch-dependent approximation that doesn't reflect the actual epoch-level macro F1. It's fine as a monitoring signal but shouldn't be confused with the real evaluation metric.

---

## 10. No reproducibility controls in training

Training scripts don't set `tf.random.set_seed()`, `np.random.seed()`, or `random.seed()`. Data generation does (`generate.py`), but training runs are non-reproducible. On GPU, full reproducibility also requires `tf.config.experimental.enable_op_determinism()`.

---

## 11. `augment_features` is pure Python loop — slow

`generate.py`:
```python
means = np.array([window[max(0, i - 4) : i + 1].mean(axis=0) for i in range(window.shape[0])])
```

This runs a Python loop per timestep per window. With ~1.5M windows, this is the data generation bottleneck. Vectorize with `np.lib.stride_tricks` or `pd.rolling`.

---

## 12. No input validation at inference boundaries

`predict.py` and `app.py` don't validate that input feature dimensions match model expectations, don't check for NaN/Inf in inputs, and don't clamp outputs. In a medical context, silent wrong answers are worse than errors.

---

## Summary — highest-leverage changes in order

| Priority | Change | Impact |
|----------|--------|--------|
| 1 | Default to balanced sampling for horizon-labeled models | Fixes forecast/forecast30 F1 from ~0.24/0.33 to ~0.99 |
| 2 | Make focal_loss alpha per-class | Proper loss weighting for remaining imbalance |
| 3 | Patient-level train/test split | Honest metrics (current ones are inflated by data leakage) |
| 4 | Save and load the scaler | Correct inference on new/live data |
| 5 | Fix live simulator feature augmentation | Working demo for 20-feature models |
| 6 | Increase EarlyStopping patience | Model actually benefits from LR reduction |
| 7 | Add clinical metrics to evaluation | Sensitivity@specificity, detection latency |
| 8 | Shuffle before validation split | Unbiased validation during training |

Items 1-3 are the ones that would most change actual model quality and trustworthiness. Item 1 alone would likely bring forecast and forecast30 in line with pretrain's 0.987 macro F1.
