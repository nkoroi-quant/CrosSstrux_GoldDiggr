"""
Regime classification helpers.

Version 2 preserves the original V1 behavior so the training tests remain stable:
- impulse == 1 -> expansion
- balance == 1 -> compression
- otherwise -> neutral

The higher-level market regime used by the inference engine is still driven by
the trained XAUUSD metadata thresholds.
"""

from __future__ import annotations

import pandas as pd


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = "unknown"

    df.loc[df["impulse"] == 1, "regime"] = "expansion"
    df.loc[df["balance"] == 1, "regime"] = "compression"
    df.loc[(df["impulse"] == 0) & (df["balance"] == 0), "regime"] = "neutral"

    return df
