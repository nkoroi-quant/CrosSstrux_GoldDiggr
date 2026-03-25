"""
Key level detection for CrossStrux v2.

This module adds the extra structural context required by GoldDiggr:
- swing highs / lows
- breakout detection
- liquidity sweep detection
- simple order-block proxy
"""

from __future__ import annotations

import pandas as pd


def detect_key_levels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Add key level columns to the dataframe.
    """
    df = df.copy()

    rolling_high = df["high"].rolling(lookback, min_periods=1).max()
    rolling_low = df["low"].rolling(lookback, min_periods=1).min()

    prev_high = rolling_high.shift(1)
    prev_low = rolling_low.shift(1)

    df["key_high"] = prev_high.fillna(df["high"])
    df["key_low"] = prev_low.fillna(df["low"])

    df["midpoint"] = (df["key_high"] + df["key_low"]) / 2.0

    # Breakout proxies
    df["breakout_up"] = ((df["close"] > df["key_high"]) & (df["impulse_dir"] > 0)).astype(int)
    df["breakout_down"] = ((df["close"] < df["key_low"]) & (df["impulse_dir"] < 0)).astype(int)

    # Liquidity sweep proxies
    sweep_up = (df["high"] > df["key_high"]) & (df["close"] < df["key_high"])
    sweep_down = (df["low"] < df["key_low"]) & (df["close"] > df["key_low"])
    df["liquidity_sweep_up"] = sweep_up.astype(int)
    df["liquidity_sweep_down"] = sweep_down.astype(int)

    # Order block proxy: a strong candle closing in direction of impulse near a level.
    df["bullish_order_block"] = (
        (df["impulse_dir"] > 0) & (df["close"] > df["open"]) & (df["close"] >= df["midpoint"])
    ).astype(int)
    df["bearish_order_block"] = (
        (df["impulse_dir"] < 0) & (df["close"] < df["open"]) & (df["close"] <= df["midpoint"])
    ).astype(int)

    return df


def summarize_key_levels(latest: pd.Series) -> dict:
    """
    Convert the last row into a small, display-friendly key level summary.
    """
    return {
        "key_high": float(latest.get("key_high", latest["high"])),
        "key_low": float(latest.get("key_low", latest["low"])),
        "midpoint": float(latest.get("midpoint", (latest["high"] + latest["low"]) / 2.0)),
        "breakout_up": bool(int(latest.get("breakout_up", 0))),
        "breakout_down": bool(int(latest.get("breakout_down", 0))),
        "liquidity_sweep_up": bool(int(latest.get("liquidity_sweep_up", 0))),
        "liquidity_sweep_down": bool(int(latest.get("liquidity_sweep_down", 0))),
        "bullish_order_block": bool(int(latest.get("bullish_order_block", 0))),
        "bearish_order_block": bool(int(latest.get("bearish_order_block", 0))),
    }
