"""
Volatility expansion detection for CrossStrux v2.
"""

from __future__ import annotations

import pandas as pd


def volatility_expansion(df: pd.DataFrame, short_window: int = 5, mid_window: int = 50) -> pd.DataFrame:
    """
    Detect volatility expansion relative to a longer baseline.
    """
    df = df.copy()

    ret = df["close"].pct_change()
    df["vol_short_std"] = ret.rolling(short_window, min_periods=1).std()
    df["vol_mid_std"] = ret.rolling(mid_window, min_periods=1).std()

    df["volatility_expansion"] = df["vol_short_std"] / (df["vol_mid_std"] + 1e-12)
    df["volatility_expansion"] = df["volatility_expansion"].rolling(3, min_periods=1).mean()

    df["volatility_expansion"] = df["volatility_expansion"].replace([pd.NA, float("inf"), float("-inf")], 0).fillna(0)
    return df
