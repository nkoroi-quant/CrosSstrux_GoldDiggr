import pandas as pd
import numpy as np

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=period, min_periods=1).mean()
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = compute_atr(df)
    close_open = df["close"] - df["open"]
    atr = df["atr"].replace(0, 1e-8)
    df["impulse"] = (close_open / atr > 1.0).astype(int)
    df["balance"] = (close_open / atr < -1.0).astype(int)
    df["impulse_norm"] = np.abs(close_open) / (df["high"] - df["low"] + 1e-8)
    df["cdi"] = (df["close"] - df["close"].rolling(20).mean()) / (df["atr"] + 1e-8)
    df["volatility_expansion"] = (df["high"] - df["low"]) / (df["atr"] + 1e-8)
    return df