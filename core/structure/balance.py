"""
Balance / compression detection for CrossStrux v2.
"""

from __future__ import annotations

import pandas as pd


def detect_balance(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Detect compression / balance regime based on volatility contraction.
    """
    df = df.copy()
    df["rolling_std"] = df["close"].rolling(window, min_periods=1).std()
    median_vol = df["rolling_std"].median()
    df["balance"] = (df["rolling_std"] < median_vol).astype(int)
    df["balance"] = df["balance"].fillna(0).astype(int)
    return df
