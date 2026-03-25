"""
Regime stability scoring.
"""

from __future__ import annotations

import pandas as pd


def regime_stability(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Measure how persistent the regime is over time.
    """
    df = df.copy()

    if "regime" not in df.columns:
        raise ValueError("regime column missing before regime_stability()")

    same_regime = (df["regime"] == df["regime"].shift()).astype(int)

    df["regime_stability"] = same_regime.rolling(window, min_periods=1).mean()
    df["regime_stability"] = df["regime_stability"].fillna(0.0)

    return df
