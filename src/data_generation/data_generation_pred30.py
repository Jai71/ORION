import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Global Constants
NUM_PATIENTS = 500         # Number of patients to simulate
TIMESTEPS = 3000           # Number of time steps per patient (1 reading per second, ~50 minutes)
FEATURES = 5               # Vital signs: HR, BP, SpO₂, RR, and Body Temperature
ANOMALY_PROBABILITY = 0.3  # Approximately 30% of patients will have an anomaly
ANOMALY_TYPES = ['anaphylaxis', 'malignant_hyperthermia', 'respiratory_depression']

# Mapping anomaly types to integer codes for windowed dataset
TYPE_TO_CODE = {
    'none': 0,
    'anaphylaxis': 1,
    'malignant_hyperthermia': 2,
    'respiratory_depression': 3
}
WINDOW = 20  # Window length in seconds for LSTM training
# Prediction horizon in seconds: label window as positive if an anomaly onset occurs within this future window
PREDICTION_HORIZON = 30

def generate_patient_data(patient_id, timesteps=TIMESTEPS, anomaly_type=None):
    """
    Generate a time-series for one patient.

    Each patient gets unique baseline values:
      - Heart Rate (HR): between 65 and 85 bpm
      - Blood Pressure (BP): between 110 and 130 mmHg
      - Oxygen Saturation (SpO₂): between 96 and 100%
      - Respiratory Rate (RR): between 12 and 16 breaths per minute
      - Body Temperature (Temp): between 36.5 and 37.5 °C

    Normal readings include small random noise. If an anomaly is to be injected,
    an onset time is chosen between 20% and 50% of the sequence. From that time
    onward, the vital signs are altered to simulate the corresponding surgical complication.

    For respiratory_depression, the anomaly is modeled in two phases:
      • Onset Phase (0–5 minutes post exposure):
            - HR: declines gradually from baseline to about a 5–10% decrease.
            - BP: declines gradually to about a 5–10% drop from baseline.
            - SpO₂: remains near baseline for the first minute, then drops so that by 5 minutes
                    it is reduced by roughly 10–13% from baseline.
            - RR: remains largely unchanged.
            - Temp: remains largely unchanged.
      • Rapid Progression Phase (5–10 minutes post exposure):
            - HR: remains near the onset level (5–10% below baseline).
            - BP: remains constant at the depressed level.
            - SpO₂: drops further to reach a total decrease of roughly 15–20% from baseline.
            - RR: remains stable.
            - Temp: remains stable.
      • After 10 minutes post exposure, values remain constant.
    """
    # Generate unique baseline values for the patient
    baseline_hr = random.uniform(65, 85)
    baseline_bp = random.uniform(110, 130)
    baseline_spo2 = random.uniform(96, 100)
    baseline_temp = random.uniform(36.5, 37.5)
    baseline_rr   = random.uniform(12, 16)  # Respiratory rate baseline

    # Create normal time-series data with small noise for all timesteps
    time_series = []
    for t in range(timesteps):
        hr = baseline_hr + np.random.normal(0, 1)
        bp = baseline_bp + np.random.normal(0, 2)
        spo2 = baseline_spo2 + np.random.normal(0, 0.5)
        rr   = baseline_rr   + np.random.normal(0, 0.5)
        temp = baseline_temp + np.random.normal(0, 0.2)
        time_series.append([hr, bp, spo2, rr, temp])

    label = 0
    anomaly_used = 'none'

    if anomaly_type is not None:
        label = 1
        anomaly_used = anomaly_type
        # Choose anomaly onset between 20% and 50% of the sequence
        onset = random.randint(timesteps // 5, timesteps // 2)

        if anomaly_type == 'anaphylaxis':
            # (Anaphylaxis anomaly code remains unchanged from previous version)
            drop_onset = random.uniform(0.20, 0.40)        # BP drop fraction during onset
            spo2_drop_onset = random.uniform(0.05, 0.10)      # SpO₂ drop fraction during onset
            delta_temp_onset = random.uniform(0.1, 0.2)       # Temp increase during onset
            incr_rr_onset   = random.uniform(0.20, 0.40)
            incr_rr_overall = random.uniform(0.30, 0.60)

            overall_hr_factor = random.uniform(1.30, 1.50)    # Overall HR multiplier at end of escalation
            overall_bp_drop = drop_onset + random.uniform(0, 0.05)  # Overall BP drop fraction
            overall_spo2_drop = random.uniform(0.10, 0.15)    # Overall SpO₂ drop fraction
            delta_temp2 = random.uniform(0, 0.1)              # Additional Temp increase during escalation

            for t in range(onset, timesteps):
                time_since_onset = t - onset  # in seconds
                if time_since_onset <= 300:
                    # Onset Phase: 0–5 minutes post exposure
                    ratio = time_since_onset / 300.0
                    new_hr = baseline_hr * (1 + 0.30 * ratio)
                    new_bp = baseline_bp * (1 - drop_onset * ratio)
                    new_spo2 = baseline_spo2 * (1 - spo2_drop_onset * ratio)
                    new_temp = baseline_temp + delta_temp_onset * ratio
                    new_rr   = baseline_rr * (1 + incr_rr_onset * ratio)
                elif time_since_onset <= 600:
                    # Rapid Escalation Phase: 5–10 minutes post exposure
                    ratio = (time_since_onset - 300) / 300.0
                    new_hr = baseline_hr * (1.30 + (overall_hr_factor - 1.30) * ratio)
                    new_bp = baseline_bp * ((1 - drop_onset) - ((overall_bp_drop - drop_onset) * ratio))
                    new_spo2 = baseline_spo2 * ((1 - spo2_drop_onset) - ((overall_spo2_drop - spo2_drop_onset) * ratio))
                    new_temp = baseline_temp + delta_temp_onset + delta_temp2 * ratio
                    new_rr   = baseline_rr * (1 + incr_rr_onset + (incr_rr_overall - incr_rr_onset) * ratio)
                else:
                    # After 10 minutes post exposure, values remain constant
                    new_hr = baseline_hr * overall_hr_factor
                    new_bp = baseline_bp * (1 - overall_bp_drop)
                    new_spo2 = baseline_spo2 * (1 - overall_spo2_drop)
                    new_temp = baseline_temp + delta_temp_onset + delta_temp2
                    new_rr   = baseline_rr * (1 + incr_rr_overall)

                time_series[t][0] = new_hr + np.random.normal(0, 1)
                time_series[t][1] = new_bp + np.random.normal(0, 2)
                time_series[t][2] = new_spo2 + np.random.normal(0, 0.5)
                time_series[t][3] = new_rr + np.random.normal(0, 0.5)
                time_series[t][4] = new_temp + np.random.normal(0, 0.2)

        elif anomaly_type == 'malignant_hyperthermia':
            # (Malignant hyperthermia anomaly code remains unchanged from previous version)
            drop_onset = random.uniform(0.10, 0.20)
            spO2_drop_onset = random.uniform(0.05, 0.07)
            spO2_drop_total = random.uniform(0.10, 0.15)
            delta_temp1 = random.uniform(0.5, 1.0)
            delta_temp2 = random.uniform(1.0, 1.5)
            rr_onset = random.uniform(0.15, 0.40)
            rr_total = random.uniform(0.40, 0.60)
            for t in range(onset, timesteps):
                time_since_onset = t - onset
                if time_since_onset <= 300:
                    ratio = time_since_onset / 300.0
                    new_hr = baseline_hr * (1 + 0.4 * ratio)
                    new_bp = baseline_bp * (1 - drop_onset * ratio)
                    new_spo2 = baseline_spo2 * (1 - spO2_drop_onset * ratio)
                    new_temp = baseline_temp + delta_temp1 * ratio
                    new_rr   = baseline_rr * (1 + rr_onset * ratio)
                elif time_since_onset <= 600:
                    ratio = (time_since_onset - 300) / 300.0
                    new_hr = baseline_hr * (1.4 + 0.2 * ratio)
                    new_bp = baseline_bp * (1 - drop_onset)
                    new_spo2 = baseline_spo2 * (1 - (spO2_drop_onset + (spO2_drop_total - spO2_drop_onset) * ratio))
                    new_temp = baseline_temp + delta_temp1 + delta_temp2 * ratio
                    new_rr   = baseline_rr * (1 + rr_onset + (rr_total - rr_onset) * ratio)
                else:
                    new_hr = baseline_hr * 1.6
                    new_bp = baseline_bp * (1 - drop_onset)
                    new_spo2 = baseline_spo2 * (1 - spO2_drop_total)
                    if new_temp > 39.0:
                        new_rr = baseline_rr * random.uniform(1.6, 2.0)
                    else:
                        new_rr = baseline_rr * (1 + rr_total)

                time_series[t][0] = new_hr + np.random.normal(0, 1)
                time_series[t][1] = new_bp + np.random.normal(0, 2)
                time_series[t][2] = new_spo2 + np.random.normal(0, 0.5)
                time_series[t][3] = new_rr + np.random.normal(0, 0.5)
                time_series[t][4] = new_temp + np.random.normal(0, 0.2)

        elif anomaly_type == 'respiratory_depression':
            # Pre-calculate parameters for respiratory depression anomaly:
            hr_drop_onset = random.uniform(0.05, 0.10)  # 5–10% drop in HR during onset phase
            bp_drop_onset = random.uniform(0.05, 0.10)  # 5–10% drop in BP during onset phase
            # For SpO₂, we want a significant drop:
            overall_spo2_drop = random.uniform(0.15, 0.20)  # total drop of 15–20% by 10 minutes
            # For the onset phase, we model a delayed drop in SpO₂:
            spo2_drop_onset = random.uniform(0.10, 0.13)  # by 5 minutes, a drop of ~10–13%
            
            for t in range(onset, timesteps):
                time_since_onset = t - onset  # in seconds
                if time_since_onset <= 300:
                    # Onset Phase (0–5 minutes)
                    ratio = time_since_onset / 300.0
                    new_hr = baseline_hr * (1 - hr_drop_onset * ratio)
                    new_bp = baseline_bp * (1 - bp_drop_onset * ratio)
                    # For SpO₂, remain near baseline for the first minute, then drop linearly
                    if time_since_onset < 60:
                        spo2_ratio = 0
                    else:
                        spo2_ratio = (time_since_onset - 60) / (300 - 60)
                    new_spo2 = baseline_spo2 * (1 - spo2_drop_onset * spo2_ratio)
                    new_rr = baseline_rr  # remains stable
                    new_temp = baseline_temp  # remains stable
                elif time_since_onset <= 600:
                    # Rapid Progression Phase (5–10 minutes)
                    ratio = (time_since_onset - 300) / 300.0
                    # HR and BP remain at the onset level
                    new_hr = baseline_hr * (1 - hr_drop_onset)
                    new_bp = baseline_bp * (1 - bp_drop_onset)
                    # SpO₂ drops further from the onset level to the overall drop level
                    new_spo2 = baseline_spo2 * (1 - (spo2_drop_onset + (overall_spo2_drop - spo2_drop_onset) * ratio))
                    new_rr = baseline_rr  # remains stable
                    new_temp = baseline_temp
                else:
                    # After 10 minutes post exposure, values remain constant
                    new_hr = baseline_hr * (1 - hr_drop_onset)
                    new_bp = baseline_bp * (1 - bp_drop_onset)
                    new_spo2 = baseline_spo2 * (1 - overall_spo2_drop)
                    new_rr = baseline_rr  # remains stable
                    new_temp = baseline_temp

                time_series[t][0] = new_hr + np.random.normal(0, 1)
                time_series[t][1] = new_bp + np.random.normal(0, 2)
                time_series[t][2] = new_spo2 + np.random.normal(0, 0.5)
                time_series[t][3] = new_rr + np.random.normal(0, 0.5)
                time_series[t][4] = new_temp + np.random.normal(0, 0.2)

    data = np.array(time_series)
    # At the end of the function, return the onset index (or None)
    return data, label, anomaly_used, onset if anomaly_type is not None else None

def generate_dataset(num_patients=NUM_PATIENTS, timesteps=TIMESTEPS):
    """
    Generates synthetic vital sign data for multiple patients.

    Returns:
        data_list: list of numpy arrays, one per patient (each shape: timesteps x 5)
        labels: list of anomaly labels (0 = normal, 1 = anomaly)
        anomaly_types: list of anomaly type strings for each patient ('none' if normal)
    """
    data_list = []
    labels = []
    anomaly_types = []
    onsets = []

    for patient_id in range(num_patients):
        # Decide whether this patient will have an anomaly
        if random.random() < ANOMALY_PROBABILITY:
            anomaly_type = random.choice(ANOMALY_TYPES)
        else:
            anomaly_type = None

        data, label, anomaly_used, onset = generate_patient_data(patient_id, timesteps, anomaly_type)
        data_list.append(data)
        labels.append(label)
        anomaly_types.append(anomaly_used)
        onsets.append(onset)

    return data_list, labels, anomaly_types, onsets


def main():
    # Generate data for all patients
    data_list, labels, anomaly_types, onsets = generate_dataset()
    print(f"Generated synthetic data for {len(data_list)} patients with {TIMESTEPS} readings each.")

    # Build windowed dataset with prediction horizon logic
    X_windows, y_codes = [], []
    for series, anomaly_type, onset in zip(data_list, anomaly_types, onsets):
        for start in range(0, TIMESTEPS - WINDOW - PREDICTION_HORIZON + 1):
            # extract past WINDOW seconds
            window = series[start : start + WINDOW]
            # label is the anomaly code if onset occurs within the next PREDICTION_HORIZON
            if onset is not None and start + WINDOW <= onset < start + WINDOW + PREDICTION_HORIZON:
                code = TYPE_TO_CODE[anomaly_type]
            else:
                code = TYPE_TO_CODE['none']
            X_windows.append(window)
            y_codes.append(code)
    X_array = np.stack(X_windows)  # shape (N, WINDOW, FEATURES)
    y_codes = np.array(y_codes)    # shape (N,)

    # Scale features
    n_windows = X_array.shape[0]
    flat = X_array.reshape(-1, FEATURES)  # shape (N * WINDOW, FEATURES)
    scaler = StandardScaler().fit(flat)
    X_scaled_flat = scaler.transform(flat)
    X_scaled = X_scaled_flat.reshape(n_windows, WINDOW, FEATURES)

    # Prepare LSTM training data (ignore respiratory_depression windows)
    mask = y_codes != TYPE_TO_CODE['respiratory_depression']
    X_lstm = X_scaled[mask]
    y_lstm_int = y_codes[mask]
    y_lstm = to_categorical(y_lstm_int, num_classes=3)

    # Save numpy arrays ready for model training (prediction dataset)
    np.save('data/processed/X_pred30_train.npy', X_lstm)
    np.save('data/processed/y_pred30_train.npy', y_lstm)
    np.save('data/processed/y_pred30_labels_full.npy', y_codes)
    print(f"Saved data/processed/X_pred30_train.npy with shape {X_lstm.shape}")
    print(f"Saved data/processed/y_pred30_train.npy with shape {y_lstm.shape}")
    print(f"Saved data/processed/y_pred30_labels_full.npy with shape {y_codes.shape}")

if __name__ == "__main__":
    main()
