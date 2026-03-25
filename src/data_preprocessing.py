# ============================================================
#  src/data_preprocessing.py
#  Tesla Stock — Data Loading, Cleaning, Scaling, Sequences
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler


# ──────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────

def load_data(path):
    """Load CSV and set Date as index."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Date range: {df.index.min().date()} → {df.index.max().date()}")
    return df


# ──────────────────────────────────────────────────────────
# 2. EXPLORE DATA
# ──────────────────────────────────────────────────────────

def explore_data(df):
    """Print shape, dtypes, stats, and missing values."""
    print("\n── Shape ──────────────────────────────────")
    print(df.shape)
    print("\n── Data Types ─────────────────────────────")
    print(df.dtypes)
    print("\n── Descriptive Statistics ─────────────────")
    print(df.describe().round(2))
    print("\n── Missing Values ──────────────────────────")
    print(df.isnull().sum())
    print(f"   Total missing: {df.isnull().sum().sum()}")


# ──────────────────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ──────────────────────────────────────────────────────────

def handle_missing(df):
    """
    Forward-fill then back-fill.
    WHY: Stock data is time-ordered. Forward-fill carries the
    last known price forward (most realistic). Mean/median
    imputation would break the time structure.
    """
    total = df.isnull().sum().sum()
    if total == 0:
        print("✅ No missing values found.")
        return df
    print(f"⚠  Found {total} missing values — applying forward-fill + back-fill")
    df = df.ffill().bfill()
    print("✅ Missing values handled.")
    return df


# ──────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────

def add_features(df):
    """Add technical indicator features."""
    df = df.copy()
    df["MA_20"]        = df["Adj Close"].rolling(20).mean()   # 20-day moving avg
    df["MA_50"]        = df["Adj Close"].rolling(50).mean()   # 50-day moving avg
    df["Daily_Return"] = df["Adj Close"].pct_change()         # % daily change
    df["Volatility"]   = df["Daily_Return"].rolling(20).std() # rolling volatility
    df.dropna(inplace=True)
    print(f"✅ Features added. New shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────
# 5. EDA PLOTS
# ──────────────────────────────────────────────────────────

def plot_eda(df, save_dir="reports"):
    """Generate and save EDA charts."""
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib.dates as mdates

    # Plot 1: Closing price + MAs
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["Adj Close"], color="steelblue",  linewidth=1,   label="Adj Close")
    ax.plot(df.index, df["MA_20"],     color="orange",     linewidth=1,   linestyle="--", label="MA 20")
    ax.plot(df.index, df["MA_50"],     color="red",        linewidth=1,   linestyle="--", label="MA 50")
    ax.set_title("Tesla Adj Close Price + Moving Averages")
    ax.set_ylabel("Price (USD)"); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_price_ma.png", dpi=150)
    plt.show()

    # Plot 2: Volume
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.bar(df.index, df["Volume"], color="slategray", alpha=0.7)
    ax.set_title("Daily Trading Volume")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/02_volume.png", dpi=150)
    plt.show()

    # Plot 3: Daily return distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Daily_Return"].dropna(), bins=80, kde=True, color="teal", ax=ax)
    ax.set_title("Daily Return Distribution")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/03_daily_return.png", dpi=150)
    plt.show()

    # Plot 4: Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    cols = ["Open","High","Low","Close","Adj Close","Volume"]
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/04_correlation.png", dpi=150)
    plt.show()

    print(f"✅ EDA plots saved to '{save_dir}/'")


# ──────────────────────────────────────────────────────────
# 6. SCALE DATA
# ──────────────────────────────────────────────────────────

def scale_data(df, feature="Adj Close"):
    """MinMax scale to range [0,1]. Returns scaled array + fitted scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = df[[feature]].values
    scaled = scaler.fit_transform(values)
    print(f"✅ Data scaled. Shape: {scaled.shape}")
    return scaled, scaler


# ──────────────────────────────────────────────────────────
# 7. CREATE SEQUENCES
# ──────────────────────────────────────────────────────────

def create_sequences(scaled, window=60, horizon=1):
    """
    Build sliding window sequences for RNN/LSTM input.

    Parameters
    ----------
    scaled  : numpy array shape (N, 1)
    window  : how many past days to use as input (default 60)
    horizon : how many future days to predict (1, 5, or 10)

    Returns
    -------
    X : shape (samples, window, 1)
    y : shape (samples,) for horizon=1  OR  (samples, horizon) for multi-step
    """
    X, y = [], []
    for i in range(window, len(scaled) - horizon + 1):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i:i + horizon, 0])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    if horizon == 1:
        y = y.ravel()
    return X, y


# ──────────────────────────────────────────────────────────
# 8. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────

def train_test_split_ts(X, y, test_ratio=0.20):
    """Chronological split — NO shuffling for time-series."""
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ──────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data("../data/TSLA.csv")
    explore_data(df)
    df = handle_missing(df)
    df = add_features(df)
    plot_eda(df)
    scaled, scaler = scale_data(df)
    for h in [1, 5, 10]:
        X, y = create_sequences(scaled, window=60, horizon=h)
        Xtr, Xte, ytr, yte = train_test_split_ts(X, y)
        print(f"Horizon {h:2d}d → Train: {Xtr.shape}, Test: {Xte.shape}")
