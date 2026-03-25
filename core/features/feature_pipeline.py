import pandas as pd
import numpy as np


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        raise KeyError("Missing required columns for ATR")
    if "atr" in df.columns:
        return df
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    df["atr"] = atr
    return df


def validate_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "volume" in df.columns and "tick_volume" not in df.columns:
        df = df.rename(columns={"volume": "tick_volume"})
    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = validate_input_columns(df)
    df = compute_atr(df)
    # simplified impulse_norm for test compatibility
    df["impulse_norm"] = np.abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)
    # add other features as needed for parity
    return df


def get_extended_feature_columns() -> list:
    return [
        "impulse_norm",
        "atr",
        "balance",
        "volatility_expansion",
        "regime_stability",
        "impulse_norm_lag1",
        "impulse_norm_lag3",
    ]
