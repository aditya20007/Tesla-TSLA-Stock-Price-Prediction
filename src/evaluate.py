# ============================================================
#  src/evaluate.py
#  Load Trained Models → Metrics → Plots → Comparison Table
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (
    load_data, handle_missing, add_features,
    scale_data, create_sequences, train_test_split_ts
)

DATA_PATH  = "../data/TSLA.csv"
MODELS_DIR = "../models"
REPORTS    = "../reports"
WINDOW     = 60
HORIZONS   = [1, 5, 10]
TEST_RATIO = 0.20

os.makedirs(REPORTS, exist_ok=True)


# ──────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-9, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compute_metrics(y_true, y_pred, label):
    if y_true.ndim > 1: y_true = y_true[:, 0]
    if y_pred.ndim > 1: y_pred = y_pred[:, 0]
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mpe  = mape(y_true, y_pred)
    print(f"[{label}]")
    print(f"  MSE  = {mse:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  MAE  = {mae:.6f}")
    print(f"  MAPE = {mpe:.2f}%")
    return {"Label": label, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE%": mpe}


def inv_scale(arr, scaler):
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    return scaler.inverse_transform(arr[:, :1]).ravel()


# ──────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────

def plot_actual_vs_pred(actual, pred_rnn, pred_lstm, horizon):
    n = min(300, len(actual))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual[-n:],    label="Actual",    color="black",      linewidth=1.2)
    ax.plot(pred_rnn[-n:],  label="SimpleRNN", color="dodgerblue", linewidth=1.0, alpha=0.85)
    ax.plot(pred_lstm[-n:], label="LSTM",      color="tomato",     linewidth=1.0, alpha=0.85)
    ax.set_title(f"Actual vs Predicted — Horizon {horizon}d (USD)")
    ax.set_xlabel("Test Sample"); ax.set_ylabel("Price (USD)")
    ax.legend(); plt.tight_layout()
    path = f"{REPORTS}/pred_h{horizon}.png"
    plt.savefig(path, dpi=150); plt.show()
    print(f"✅ Saved: {path}")


def plot_loss(rnn_hist, lstm_hist, horizon):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, hist, name in zip(axes, [rnn_hist, lstm_hist], ["SimpleRNN", "LSTM"]):
        ax.plot(hist.history["loss"],     label="Train Loss")
        ax.plot(hist.history["val_loss"], label="Val Loss",  linestyle="--")
        ax.set_title(f"{name} Loss — Horizon {horizon}d")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend()
    plt.tight_layout()
    path = f"{REPORTS}/loss_h{horizon}.png"
    plt.savefig(path, dpi=150); plt.show()
    print(f"✅ Saved: {path}")


def plot_comparison(all_metrics):
    df = pd.DataFrame(all_metrics)
    pivot = df.pivot(index="Horizon", columns="Model", values="RMSE")
    ax = pivot.plot(kind="bar", figsize=(10, 5), color=["dodgerblue", "tomato"])
    ax.set_title("RMSE Comparison: SimpleRNN vs LSTM")
    ax.set_xlabel("Forecast Horizon (days)"); ax.set_ylabel("RMSE (scaled)")
    ax.legend(title="Model"); plt.xticks(rotation=0)
    plt.tight_layout()
    path = f"{REPORTS}/comparison.png"
    plt.savefig(path, dpi=150); plt.show()
    print(f"✅ Saved: {path}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    all_metrics = []

    for h in HORIZONS:
        print(f"\n{'='*50}")
        print(f"  Evaluating — Horizon {h}d")
        print(f"{'='*50}")

        df = load_data(DATA_PATH)
        df = handle_missing(df)
        df = add_features(df)
        scaled, scaler = scale_data(df)
        X, y = create_sequences(scaled, WINDOW, h)
        _, Xte, _, yte = train_test_split_ts(X, y, TEST_RATIO)

        rnn_path  = f"{MODELS_DIR}/SimpleRNN_h{h}_final.h5"
        lstm_path = f"{MODELS_DIR}/LSTM_h{h}_final.h5"

        if not os.path.exists(rnn_path) or not os.path.exists(lstm_path):
            print(f"⚠  Models not found for h={h}. Run train.py first.")
            continue

        rnn_model  = load_model(rnn_path,  compile=False)
        lstm_model = load_model(lstm_path, compile=False)

        pred_rnn  = rnn_model.predict(Xte,  verbose=0)
        pred_lstm = lstm_model.predict(Xte, verbose=0)

        # Metrics in scaled space
        m_rnn  = compute_metrics(yte, pred_rnn,  f"SimpleRNN h={h}")
        m_lstm = compute_metrics(yte, pred_lstm, f"LSTM      h={h}")
        all_metrics.append({**m_rnn,  "Horizon": h, "Model": "SimpleRNN"})
        all_metrics.append({**m_lstm, "Horizon": h, "Model": "LSTM"})

        # USD plots
        actual_usd = inv_scale(yte,       scaler)
        rnn_usd    = inv_scale(pred_rnn,  scaler)
        lstm_usd   = inv_scale(pred_lstm, scaler)
        plot_actual_vs_pred(actual_usd, rnn_usd, lstm_usd, h)

    # Summary
    if all_metrics:
        df_m = pd.DataFrame(all_metrics)
        df_m.to_csv(f"{REPORTS}/all_metrics.csv", index=False)
        print(f"\n✅ Metrics saved: {REPORTS}/all_metrics.csv")
        print(df_m[["Model","Horizon","RMSE","MAE","MAPE%"]].to_string(index=False))
        plot_comparison(all_metrics)


if __name__ == "__main__":
    main()
