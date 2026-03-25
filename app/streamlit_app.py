# ============================================================
#  app/streamlit_app.py
#  Tesla Stock Price Prediction — Streamlit Dashboard
#
#  Run: streamlit run app/streamlit_app.py
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

# ── Path setup (works both locally and on Streamlit Cloud) ──
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


def load_keras_model(name, h):
    """Load a Keras .h5 model. Returns None if TensorFlow is unavailable."""
    try:
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(
            os.path.join(MODELS_DIR, f"{name}_h{h}_final.h5"),
            compile=False
        )
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/b/bb/Tesla_T_symbol.svg",
    width=60
)
st.sidebar.title("⚙️ Controls")

horizon      = st.sidebar.selectbox("📅 Forecast Horizon", [1, 5, 10], index=0,
                                    format_func=lambda x: f"{x} Day{'s' if x>1 else ''}")
model_choice = st.sidebar.radio("🤖 Model", ["SimpleRNN", "LSTM", "Both"])
show_ma      = st.sidebar.checkbox("📈 Show Moving Averages", value=True)

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

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Days",    f"{len(df):,}")
    c2.metric("Start Date",    str(df.index.min().date()))
    c3.metric("End Date",      str(df.index.max().date()))
    c4.metric("Min Price",     f"${df['Adj Close'].min():.2f}")
    c5.metric("Max Price",     f"${df['Adj Close'].max():.2f}")

    st.markdown("---")

    # Price chart
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, df["Adj Close"], color="steelblue", linewidth=1, label="Adj Close")
    if show_ma:
        ax.plot(df.index, df["MA_20"], color="orange", linewidth=1,
                linestyle="--", label="MA 20")
        ax.plot(df.index, df["MA_50"], color="red", linewidth=1,
                linestyle="--", label="MA 50")
    ax.set_title("Tesla Adjusted Closing Price (2010–2019)")
    ax.set_ylabel("Price (USD)"); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    # Volume
    with col1:
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.bar(df.index, df["Volume"], color="slategray", alpha=0.7)
        ax2.set_title("Daily Volume")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        st.pyplot(fig2)

    # Return distribution
    with col2:
        import seaborn as sns
        fig3, ax3 = plt.subplots(figsize=(7, 3))
        sns.histplot(df["Daily_Return"].dropna(), bins=80, kde=True, color="teal", ax=ax3)
        ax3.set_title("Daily Return Distribution")
        st.pyplot(fig3)

    # Correlation heatmap
    st.markdown("#### Feature Correlation")
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    import seaborn as sns
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    # Raw data
    st.markdown("#### Raw Data (last 20 rows)")
    st.dataframe(df.tail(20), use_container_width=True)


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

    # Check models exist
    missing = [m for m in models_to_show if not model_exists(m, horizon)]
    if missing:
        st.warning(
            f"⚠️ Model(s) not trained yet: {missing}\n\n"
            "Run `python src/train.py` locally first, then commit the `.h5` files to GitHub."
        )
    else:
        fig, ax = plt.subplots(figsize=(13, 5))
        actual_usd = inv(yte, scaler)
        ax.plot(actual_usd, label="Actual", color="black", linewidth=1.2)

        for mname in models_to_show:
            model = load_keras_model(mname, horizon)
            if model is None:
                st.error(f"❌ Could not load model '{mname}' — TensorFlow may not be installed.")
                continue
            pred     = model.predict(Xte, verbose=0)
            pred_usd = inv(pred, scaler)
            color    = "dodgerblue" if mname == "SimpleRNN" else "tomato"
            ax.plot(pred_usd, label=mname, color=color, linewidth=1, alpha=0.85)

        ax.set_title(f"Actual vs Predicted — Horizon {horizon}d (USD)")
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════
# TAB 3 — METRICS
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Model Performance Comparison")

    rows = []
    for h in [1, 5, 10]:
        sc_h, scaler_h = scale_data(df)
        Xh, yh = create_sequences(sc_h, WINDOW, h)
        _, Xte_h, _, yte_h = train_test_split_ts(Xh, yh)
        actual_h = inv(yte_h, scaler_h)

        for mname in ["SimpleRNN", "LSTM"]:
            if not model_exists(mname, h):
                rows.append({"Model": mname, "Horizon": h,
                             "RMSE ($)": "Not trained", "MAE ($)": "-", "MAPE (%)": "-"})
                continue
            mdl = load_keras_model(mname, h)
            if mdl is None:
                rows.append({"Model": mname, "Horizon": h,
                             "RMSE ($)": "TF unavailable", "MAE ($)": "-", "MAPE (%)": "-"})
                continue
            p = inv(mdl.predict(Xte_h, verbose=0), scaler_h)
            rows.append({
                "Model":    mname,
                "Horizon":  h,
                "RMSE ($)": round(np.sqrt(mean_squared_error(actual_h, p)), 2),
                "MAE ($)":  round(mean_absolute_error(actual_h, p), 2),
                "MAPE (%)": round(mape(actual_h, p), 2),
            })

    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True)

    # Bar chart — RMSE
    try:
        num_df = metrics_df[~metrics_df["RMSE ($)"].isin(["Not trained", "TF unavailable"])].copy()
        num_df["RMSE ($)"] = num_df["RMSE ($)"].astype(float)
        pivot = num_df.pivot(index="Horizon", columns="Model", values="RMSE ($)")
        fig_b, ax_b = plt.subplots(figsize=(8, 4))
        pivot.plot(kind="bar", ax=ax_b, color=["dodgerblue", "tomato"])
        ax_b.set_title("RMSE (USD) — SimpleRNN vs LSTM per Horizon")
        ax_b.set_xlabel("Horizon (days)"); ax_b.set_ylabel("RMSE (USD)")
        plt.xticks(rotation=0)
        st.pyplot(fig_b)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# TAB 4 — FUTURE FORECAST
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔭 Predict Future Stock Prices")

    col1, col2 = st.columns([1, 2])
    with col1:
        chosen = st.selectbox("Choose Model", ["SimpleRNN", "LSTM"])
        n_days = st.slider("Days to Forecast", 1, 30, 10)
        go     = st.button("🔮 Generate Forecast", type="primary")

    if go:
        if not model_exists(chosen, 1):
            st.error(f"Model not found: {chosen}_h1. Train it first with `python src/train.py`")
        else:
            model_f = load_keras_model(chosen, 1)
            if model_f is None:
                st.error("❌ Could not load model — TensorFlow may not be installed.")
            else:
                sc_f, scaler_f = scale_data(df)
                seed = sc_f[-WINDOW:].reshape(1, WINDOW, 1)

                preds_sc = []
                current  = seed.copy()
                for _ in range(n_days):
                    p = model_f.predict(current, verbose=0)[0, 0]
                    preds_sc.append(p)
                    current = np.concatenate([current[:, 1:, :], [[[p]]]], axis=1)

                preds_usd    = scaler_f.inverse_transform(
                    np.array(preds_sc).reshape(-1, 1)
                ).ravel()
                last_date    = df.index[-1]
                future_dates = pd.bdate_range(start=last_date, periods=n_days + 1)[1:]

                fc_df = pd.DataFrame({
                    "Date": future_dates[:n_days],
                    "Predicted Price ($)": preds_usd.round(2)
                })

                with col2:
                    st.dataframe(fc_df, use_container_width=True)

                fig_fc, ax_fc = plt.subplots(figsize=(12, 4))
                recent = df["Adj Close"].tail(60)
                ax_fc.plot(recent.index, recent.values,
                           label="Historical (last 60d)", color="black", linewidth=1.2)
                ax_fc.plot(future_dates[:n_days], preds_usd,
                           label=f"{chosen} Forecast",
                           color="dodgerblue", linewidth=1.5,
                           marker="o", markersize=4)
                ax_fc.axvline(last_date, color="red", linestyle="--", alpha=0.5,
                              label="Forecast Start")
                ax_fc.set_title(f"{chosen} — {n_days}-Day Price Forecast")
                ax_fc.set_ylabel("Price (USD)"); ax_fc.legend()
                st.pyplot(fig_fc)


# ══════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════
with tab5:
    st.subheader("📖 About This Project")
    st.markdown("""
    ## Tesla Stock Price Prediction using Deep Learning

    ### Problem Statement
    Predict Tesla's adjusted closing stock price using Sequential Deep Learning models
    for **1-day**, **5-day**, and **10-day** future horizons.

    ### Dataset
    | Column | Description |
    |---|---|
    | Date | Trading date |
    | Open | Opening price |
    | High | Highest price of the day |
    | Low | Lowest price of the day |
    | Close | Closing price |
    | **Adj Close** | ✅ **Target variable** — adjusted for splits/dividends |
    | Volume | Number of shares traded |

    ### Approach
    1. **Data Loading & EDA** — understand patterns, distributions, correlations
    2. **Missing Value Handling** — forward-fill (time-series safe strategy)
    3. **Feature Engineering** — MA_20, MA_50, Daily Return, Volatility
    4. **Scaling** — MinMaxScaler [0,1] for neural network convergence
    5. **Sequence Creation** — 60-day sliding window → predict next 1/5/10 days
    6. **SimpleRNN** — baseline recurrent model
    7. **LSTM** — advanced gated recurrent model (better long-term memory)
    8. **GridSearchCV** — hyperparameter tuning (units, dropout, learning rate)
    9. **Evaluation** — MSE, RMSE, MAE, MAPE
    10. **Deployment** — Streamlit web application

    ### Why LSTM outperforms SimpleRNN?
    > SimpleRNN suffers from **vanishing gradients** — it forgets information
    > from many steps ago. LSTM solves this with **3 gates** (Forget, Input, Output)
    > that control what to remember and what to discard.

    ### Tech Stack
    `Python 3.11` | `TensorFlow 2.15` | `Keras` | `Scikit-learn` |
    `Pandas` | `NumPy` | `Matplotlib` | `Seaborn` | `Streamlit`

    ### Business Use Cases
    - 📈 Algorithmic trading strategies
    - 🛡️ Risk management & portfolio optimization
    - 💼 Long-term investment planning
    - 🔍 Competitor stock comparison
    """)