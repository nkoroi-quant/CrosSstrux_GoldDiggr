# core/features/feature_pipeline.py

"""
Feature pipeline for CrossStrux v2 (FINAL STABLE)

- Fully compatible with training + inference
- Provides BOTH feature APIs:
  - get_feature_columns (legacy)
  - get_extended_feature_columns (v2)
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


REQUIRED_OHLC = ["open", "high", "low", "close"]


# ================= INPUT VALIDATION ================= #
def validate_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "time" not in df.columns and "timestamp" in df.columns:
        df["time"] = df["timestamp"]
        df = df.drop(columns=["timestamp"])

    if "time" not in df.columns:
        raise ValueError("Missing 'time' column")

    missing = [c for c in REQUIRED_OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    if "tick_volume" not in df.columns:
        if "volume" in df.columns:
            df["tick_volume"] = df["volume"]
        elif "real_volume" in df.columns:
            df["tick_volume"] = df["real_volume"]
        else:
            df["tick_volume"] = 0.0

    return df.sort_values("time").reset_index(drop=True)


# ================= ATR ================= #
def compute_atr(df: pd.DataFrame, period: int = 14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()

    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(period, min_periods=1).mean().fillna(0)


# ================= MAIN PIPELINE ================= #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = validate_input_columns(df)
    df = df.copy()

    # ATR
    df["atr"] = compute_atr(df)
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

    # STRUCTURE SAFE DEFAULTS
    df["impulse"] = df.get("impulse", 0)
    df["balance"] = df.get("balance", 0)
    df["volatility_expansion"] = df.get("volatility_expansion", 0)
    df["regime_stability"] = df.get("regime_stability", 0)

    # IMPULSE
    df["impulse_strength"] = (
        df["close"].pct_change().abs().fillna(0)
        / (df["atr_pct"] + 1e-12)
    )

    df["impulse_norm"] = df["impulse_strength"] / (df["atr_pct"] + 1e-12)

    # LAGS
    lags = [1, 3, 5, 10]
    for lag in lags:
        df[f"impulse_norm_lag{lag}"] = df["impulse_norm"].shift(lag)
        df[f"balance_lag{lag}"] = df["balance"].shift(lag)
        df[f"volatility_expansion_lag{lag}"] = df["volatility_expansion"].shift(lag)

    # MOMENTUM
    df["impulse_momentum_5"] = df["impulse_norm"].diff(5)
    df["balance_momentum_5"] = df["balance"].diff(5)
    df["volatility_momentum_5"] = df["volatility_expansion"].diff(5)

    # ACCELERATION
    df["impulse_acceleration_5"] = df["impulse_momentum_5"].diff(5)

    feature_columns = get_extended_feature_columns()

    df = df[["time"] + feature_columns]

    return (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )


# ================= FEATURE SETS ================= #

def get_feature_columns() -> List[str]:
    """
    🔥 LEGACY SUPPORT (TRAINING EXPECTS THIS)
    Now maps to extended features to avoid mismatch.
    """
    return get_extended_feature_columns()


def get_extended_feature_columns() -> List[str]:
    """
    Full v2 feature set
    """
    return [
        "impulse_norm",
        "balance",
        "volatility_expansion",
        "regime_stability",
        "impulse_norm_lag1",
        "impulse_norm_lag3",
        "impulse_norm_lag5",
        "impulse_norm_lag10",
        "balance_lag1",
        "balance_lag3",
        "balance_lag5",
        "balance_lag10",
        "volatility_expansion_lag1",
        "volatility_expansion_lag3",
        "volatility_expansion_lag5",
        "volatility_expansion_lag10",
        "impulse_momentum_5",
        "balance_momentum_5",
        "volatility_momentum_5",
        "impulse_acceleration_5",
    ]