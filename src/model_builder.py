# ============================================================
#  src/model_builder.py
#  SimpleRNN and LSTM Model Architectures
# ============================================================

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SimpleRNN, LSTM, Dense, Dropout,
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)


# ──────────────────────────────────────────────────────────
# 1. SIMPLERNN MODEL
# ──────────────────────────────────────────────────────────

def build_simple_rnn(units=64, dropout_rate=0.2, learning_rate=0.001, horizon=1):
    """
    Two-layer SimpleRNN model.

    Architecture:
      Input → SimpleRNN(units) → Dropout → SimpleRNN(units/2)
            → Dropout → Dense(32, relu) → Dense(horizon)

    Parameters
    ----------
    units         : number of RNN neurons
    dropout_rate  : fraction of neurons to drop (prevents overfitting)
    learning_rate : Adam optimizer step size
    horizon       : output size (1, 5, or 10 days)
    """
    model = Sequential(name="SimpleRNN_Model")
    model.add(Input(shape=(60, 1)))
    model.add(SimpleRNN(units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(units // 2, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(horizon))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


# ──────────────────────────────────────────────────────────
# 2. LSTM MODEL
# ──────────────────────────────────────────────────────────

def build_lstm(units=64, dropout_rate=0.2, learning_rate=0.001, horizon=1):
    """
    Two-layer LSTM model with BatchNormalization.

    Architecture:
      Input → LSTM(units) → Dropout → LSTM(units/2)
            → Dropout → BatchNorm → Dense(32, relu) → Dense(horizon)

    Why LSTM over RNN:
      LSTM has 'gates' (forget/input/output) that control
      what information to keep or discard — solving the
      vanishing gradient problem of SimpleRNN for long sequences.
    """
    model = Sequential(name="LSTM_Model")
    model.add(Input(shape=(60, 1)))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units // 2, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(horizon))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


# ──────────────────────────────────────────────────────────
# 3. CALLBACKS
# ──────────────────────────────────────────────────────────

def get_callbacks(model_name, save_dir="models"):
    """
    Training callbacks:
    - EarlyStopping    : stop if val_loss doesn't improve for 15 epochs
    - ModelCheckpoint  : save the best model weights automatically
    - ReduceLROnPlateau: halve learning rate if val_loss plateaus for 7 epochs
    """
    os.makedirs(save_dir, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=f"{save_dir}/{model_name}_best.h5",
            monitor="val_loss", save_best_only=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
    ]


# ──────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    for h in [1, 5, 10]:
        print(f"\n── Horizon {h}d ──────────────────────────")
        build_simple_rnn(horizon=h).summary()
        build_lstm(horizon=h).summary()
