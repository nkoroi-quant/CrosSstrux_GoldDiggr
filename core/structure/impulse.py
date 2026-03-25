"""
Impulse detection for CrossStrux v2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_impulse(
    df: pd.DataFrame,
    atr_period: int = 14,
    burst_window: int = 5,
    atr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Detect multi-bar directional impulse bursts.

    Output columns:
        return
        atr
        atr_pct
        impulse_strength
        impulse
        impulse_dir
    """
    df = df.copy()

    df["return"] = df["close"].pct_change()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = np.maximum(high_low, np.maximum(high_close, low_close))

    df["atr"] = tr.rolling(atr_period, min_periods=1).mean()
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

    directional_move = df["close"] - df["close"].shift(burst_window)
    momentum = directional_move / df["close"].shift(burst_window)

    df["impulse_strength"] = momentum.abs() / (df["atr_pct"] * np.sqrt(burst_window) + 1e-12)

    df["impulse"] = (df["impulse_strength"] > atr_multiplier).astype(int)
    df["impulse_dir"] = np.sign(momentum).fillna(0)

    df["impulse_strength"] = df["impulse_strength"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["impulse_dir"] = df["impulse_dir"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df
