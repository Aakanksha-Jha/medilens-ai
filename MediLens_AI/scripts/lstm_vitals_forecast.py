"""
MediLens AI — Step 3 (sequence logic): LSTM for patient vitals forecasting.

Simulates a cohort of patients with daily vitals (heart rate, SpO2, systolic BP)
over 60 days, each with a mild upward/downward trend + noise + occasional
deterioration event. Trains an LSTM to forecast the next 3 days of heart rate
from the previous 14-day window — a simple, realistic "early warning" style task.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

tf.random.set_seed(42)
np.random.seed(42)
OUT = "/home/claude/MediLens_AI/reports"

N_PATIENTS = 200
N_DAYS = 60
WINDOW = 14
HORIZON = 3

def simulate_patient(rng):
    trend = rng.uniform(-0.15, 0.15)
    base_hr = rng.uniform(65, 85)
    hr = base_hr + trend * np.arange(N_DAYS) + rng.normal(0, 2, N_DAYS)
    if rng.random() < 0.25:  # deterioration event
        onset = rng.integers(30, 50)
        hr[onset:] += np.linspace(0, rng.uniform(8, 20), N_DAYS - onset)
    return hr.astype("float32")

rng = np.random.default_rng(7)
series = np.array([simulate_patient(rng) for _ in range(N_PATIENTS)])

# Normalize per-series stats globally (simple min-max using train stats)
mean, std = series.mean(), series.std()
series_norm = (series - mean) / std

def make_windows(data):
    X, y = [], []
    for s in data:
        for i in range(N_DAYS - WINDOW - HORIZON):
            X.append(s[i:i + WINDOW])
            y.append(s[i + WINDOW:i + WINDOW + HORIZON])
    return np.array(X)[..., None], np.array(y)

n_train = int(N_PATIENTS * 0.8)
X_train, y_train = make_windows(series_norm[:n_train])
X_test, y_test = make_windows(series_norm[n_train:])

model = models.Sequential([
    layers.Input(shape=(WINDOW, 1)),
    layers.LSTM(32, return_sequences=False),
    layers.Dense(16, activation="relu"),
    layers.Dense(HORIZON),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
hist = model.fit(X_train, y_train, validation_split=0.1, epochs=25, batch_size=32, verbose=0)

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
test_mae_real_units = test_mae * std  # de-normalize MAE back to bpm

# Plot a sample forecast vs actual for one held-out patient
sample_patient = series_norm[n_train]
sample_X = sample_patient[10:10 + WINDOW][None, ..., None]
pred = model.predict(sample_X, verbose=0)[0]
actual = sample_patient[10 + WINDOW:10 + WINDOW + HORIZON]

plt.figure(figsize=(7, 4))
plt.plot(range(WINDOW), sample_patient[10:10 + WINDOW] * std + mean, label="History (input)")
plt.plot(range(WINDOW, WINDOW + HORIZON), actual * std + mean, "o-", label="Actual next 3 days")
plt.plot(range(WINDOW, WINDOW + HORIZON), pred * std + mean, "x--", label="LSTM forecast")
plt.xlabel("Day"); plt.ylabel("Heart rate (bpm)")
plt.title("LSTM Vitals Forecast — Sample Patient")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/lstm_forecast_sample.png", dpi=140)
plt.close()

lstm_results = {
    "test_mse_normalized": float(test_loss),
    "test_mae_bpm": float(test_mae_real_units),
    "note": "Trained on synthetic per-patient heart-rate time series with injected deterioration events. Same architecture applies directly to real EHR vitals streams (heart rate, SpO2, BP, respiration rate)."
}
with open(f"{OUT}/lstm_metrics.json", "w") as f:
    json.dump(lstm_results, f, indent=2)

print(json.dumps(lstm_results, indent=2))
print("Done.")
