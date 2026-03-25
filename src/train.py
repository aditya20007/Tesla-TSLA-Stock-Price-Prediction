# ============================================================
#  src/train.py
#  Full Training Pipeline — SimpleRNN + LSTM + GridSearchCV
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add src/ to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import (
    load_data, handle_missing, add_features,
    scale_data, create_sequences, train_test_split_ts
)
from model_builder import build_simple_rnn, build_lstm, get_callbacks

from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor


# ──────────────────────────────────────────────────────────
# CONFIG — change these if needed
# ──────────────────────────────────────────────────────────
DATA_PATH  = "../data/TSLA.csv"
MODELS_DIR = "../models"
WINDOW     = 60       # 60 days look-back
HORIZONS   = [1, 5, 10]
EPOCHS     = 100
BATCH      = 32
TEST_RATIO = 0.20


# ──────────────────────────────────────────────────────────
# HELPER — prepare data for a given horizon
# ──────────────────────────────────────────────────────────

def prepare(horizon):
    df = load_data(DATA_PATH)
    df = handle_missing(df)
    df = add_features(df)
    scaled, scaler = scale_data(df)
    X, y = create_sequences(scaled, WINDOW, horizon)
    Xtr, Xte, ytr, yte = train_test_split_ts(X, y, TEST_RATIO)
    return Xtr, Xte, ytr, yte, scaler


# ──────────────────────────────────────────────────────────
# STEP 1 — TRAIN ONE MODEL
# ──────────────────────────────────────────────────────────

def train_one(build_fn, name, Xtr, ytr, horizon):
    """Build, train, save one model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\n🔥 Training {name} | Horizon={horizon}d")

    model = build_fn(horizon=horizon)
    model.summary()

    history = model.fit(
        Xtr, ytr,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_split=0.1,
        callbacks=get_callbacks(f"{name}_h{horizon}", MODELS_DIR),
        verbose=1
    )

    save_path = f"{MODELS_DIR}/{name}_h{horizon}_final.h5"
    model.save(save_path)
    print(f"✅ Saved: {save_path}")
    return model, history


# ──────────────────────────────────────────────────────────
# STEP 2 — GRIDSEARCHCV (LSTM, horizon=1)
# ──────────────────────────────────────────────────────────

def run_gridsearch(Xtr, ytr):
    """Tune LSTM hyperparameters using GridSearchCV."""
    print("\n🔍 Running GridSearchCV for LSTM (horizon=1)...")

    def make_model(units=64, dropout_rate=0.2, learning_rate=0.001):
        return build_lstm(
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            horizon=1
        )

    reg = KerasRegressor(
        model=make_model,
        epochs=30,
        batch_size=32,
        verbose=0
    )

    param_grid = {
        "model__units":         [32, 64],
        "model__dropout_rate":  [0.1, 0.2],
        "model__learning_rate": [0.001, 0.0005],
    }

    gs = GridSearchCV(
        estimator=reg,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        verbose=2
    )

    y_flat = ytr.ravel() if ytr.ndim > 1 else ytr
    gs.fit(Xtr, y_flat)

    print(f"\n✅ Best Params : {gs.best_params_}")
    print(f"   Best CV MSE : {-gs.best_score_:.6f}")

    results = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
    results.to_csv(f"{MODELS_DIR}/gridsearch_results.csv", index=False)
    print(f"   Results saved: {MODELS_DIR}/gridsearch_results.csv")

    return gs.best_params_


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Tesla Stock Price Prediction — Training")
    print("=" * 55)

    all_results = {}

    for h in HORIZONS:
        print(f"\n{'='*55}")
        print(f"  HORIZON = {h} day(s)")
        print(f"{'='*55}")

        Xtr, Xte, ytr, yte, scaler = prepare(h)

        # Save scaler (needed in evaluate + streamlit)
        joblib.dump(scaler, f"{MODELS_DIR}/scaler_h{h}.pkl")
        print(f"✅ Scaler saved: {MODELS_DIR}/scaler_h{h}.pkl")

        # Train SimpleRNN
        rnn, rnn_hist = train_one(build_simple_rnn, "SimpleRNN", Xtr, ytr, h)

        # Train LSTM
        lstm, lstm_hist = train_one(build_lstm, "LSTM", Xtr, ytr, h)

        all_results[h] = {
            "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte,
            "rnn": rnn, "lstm": lstm,
            "rnn_hist": rnn_hist, "lstm_hist": lstm_hist,
            "scaler": scaler
        }

    # Run GridSearchCV (only for h=1)
    print("\n" + "="*55)
    best_params = run_gridsearch(all_results[1]["Xtr"], all_results[1]["ytr"])

    print("\n✅ All training done! Models saved in:", MODELS_DIR)
    return all_results


if __name__ == "__main__":
    main()
