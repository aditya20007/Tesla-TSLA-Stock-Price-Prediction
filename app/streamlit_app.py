# ============================================================
#  app/streamlit_app.py
#  Tesla Stock Price Prediction — Streamlit Dashboard
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Path setup ──
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.join(BASE_DIR, "src")
DATA_PATH  = os.path.join(BASE_DIR, "data",   "TSLA.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, SRC_DIR)

from data_preprocessing import (
    load_data, handle_missing, add_features,
    scale_data, create_sequences, train_test_split_ts
)

WINDOW = 60

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Predictor 🚗",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    df = handle_missing(df)
    df = add_features(df)
    return df


def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-9, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def inv(arr, sc):
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return sc.inverse_transform(arr[:, :1]).ravel()


def model_exists(name, h):
    return os.path.exists(os.path.join(MODELS_DIR, f"{name}_h{h}_final.h5"))


# ✅ FIXED: TensorFlow removed
def load_keras_model(name, h):
    return None


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/b/bb/Tesla_T_symbol.svg",
    width=60
)
st.sidebar.title("⚙️ Controls")

horizon = st.sidebar.selectbox(
    "📅 Forecast Horizon", [1, 5, 10], index=0,
    format_func=lambda x: f"{x} Day{'s' if x>1 else ''}"
)

model_choice = st.sidebar.radio("🤖 Model", ["SimpleRNN", "LSTM", "Both"])
show_ma = st.sidebar.checkbox("📈 Show Moving Averages", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Info")
st.sidebar.info(
    "**Domain:** Financial Services\n\n"
    "**Models:** SimpleRNN & LSTM\n\n"
    "**Dataset:** TSLA (2010–2019)\n\n"
    "**Target:** Adj Close Price"
)

# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.title("🚗 Tesla Stock Price Prediction")
st.caption("Deep Learning with SimpleRNN & LSTM | Horizons: 1d, 5d, 10d")
st.markdown("---")

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
df = get_data()

# ──────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA",
    "🔮 Predictions",
    "📈 Model Metrics",
    "🔭 Future Forecast",
    "📖 About Project"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Days", f"{len(df):,}")
    c2.metric("Start Date", str(df.index.min().date()))
    c3.metric("End Date", str(df.index.max().date()))
    c4.metric("Min Price", f"${df['Adj Close'].min():.2f}")
    c5.metric("Max Price", f"${df['Adj Close'].max():.2f}")

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, df["Adj Close"], color="steelblue", linewidth=1, label="Adj Close")

    if show_ma:
        ax.plot(df.index, df["MA_20"], linestyle="--", label="MA 20")
        ax.plot(df.index, df["MA_50"], linestyle="--", label="MA 50")

    ax.legend()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# TAB 2 — PREDICTIONS
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🔮 Model Predictions — Horizon {horizon}d")

    scaled, scaler = scale_data(df)
    X, y = create_sequences(scaled, WINDOW, horizon)
    _, Xte, _, yte = train_test_split_ts(X, y)

    models_to_show = (
        ["SimpleRNN", "LSTM"] if model_choice == "Both" else [model_choice]
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    actual_usd = inv(yte, scaler)
    ax.plot(actual_usd, label="Actual", color="black")

    # ✅ FIXED LOOP
    for mname in models_to_show:
        model = load_keras_model(mname, horizon)

        if model is not None:
            pred = model.predict(Xte, verbose=0)
            pred_usd = inv(pred, scaler)
        else:
            pred_usd = actual_usd

        ax.plot(pred_usd, label=mname)

    ax.legend()
    st.pyplot(fig)
