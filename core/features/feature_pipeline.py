from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["time", "open", "high", "low", "close"]


def validate_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the minimum OHLC input schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)

    if "tick_volume" not in out.columns:
        if "volume" in out.columns:
            out["tick_volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
        else:
            out["tick_volume"] = 0.0
    else:
        out["tick_volume"] = pd.to_numeric(out["tick_volume"], errors="coerce").fillna(0)

    if "spread" not in out.columns:
        out["spread"] = 0.0

    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ATR using only trailing candles.

    If an ATR column already exists, preserve it unchanged so callers can reuse
    precomputed values without introducing drift.
    """
    if "atr" in df.columns:
        return df.copy()

    out = validate_input_columns(df)
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr"] = true_range.rolling(window=period, min_periods=1).mean().fillna(0.0)
    return out


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a deterministic trailing-only feature table."""
    out = compute_atr(df)

    o = out["open"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)
    v = out["tick_volume"].astype(float)
    atr = out["atr"].astype(float).replace(0, 1e-8)
    rng = (h - l).replace(0, 1e-8)

    out["range"] = rng
    out["body"] = (c - o).abs()
    out["upper_wick"] = (h - np.maximum(o, c)).clip(lower=0.0)
    out["lower_wick"] = (np.minimum(o, c) - l).clip(lower=0.0)
    out["body_to_range"] = _safe_div(out["body"], rng).fillna(0.0)
    out["close_open"] = c - o
    out["close_to_high"] = _safe_div(h - c, rng).fillna(0.0)
    out["close_to_low"] = _safe_div(c - l, rng).fillna(0.0)
    out["return_1"] = c.pct_change(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["return_3"] = c.pct_change(3).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["return_5"] = c.pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["return_10"] = c.pct_change(10).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["ma_5"] = _rolling_mean(c, 5)
    out["ma_10"] = _rolling_mean(c, 10)
    out["ma_20"] = _rolling_mean(c, 20)
    out["ma_diff_5_20"] = out["ma_5"] - out["ma_20"]
    out["trend_5"] = c - c.shift(5)
    out["trend_10"] = c - c.shift(10)
    out["volume_ma_5"] = _rolling_mean(v, 5)
    out["volume_ratio"] = _safe_div(v, out["volume_ma_5"]).fillna(0.0)
    out["atr_pct"] = _safe_div(out["atr"], c.abs().replace(0, 1e-8)).fillna(0.0)
    out["impulse"] = ((out["body"] / atr) > 0.75).astype(int)
    out["balance"] = ((out["body"] / atr) < 0.35).astype(int)
    out["impulse_norm"] = _safe_div(out["body"], rng).clip(lower=0.0).fillna(0.0)
    out["cdi"] = _safe_div(c - out["ma_20"], atr).fillna(0.0)
    out["volatility_expansion"] = _safe_div(rng, atr).fillna(0.0)
    out["rolling_range_5"] = _rolling_mean(rng, 5)
    out["rolling_range_10"] = _rolling_mean(rng, 10)
    out["rolling_range_20"] = _rolling_mean(rng, 20)
    out["breakout_up"] = (c > h.shift(1).rolling(5, min_periods=1).max().fillna(h)).astype(int)
    out["breakout_down"] = (c < l.shift(1).rolling(5, min_periods=1).min().fillna(l)).astype(int)
    out["momentum_points"] = c.diff().fillna(0.0)
    out["price_position_20"] = _safe_div(c - out["ma_20"], rng).fillna(0.0)
    out["volatility_z"] = _safe_div(rng - _rolling_mean(rng, 20), _rolling_std(rng, 20).replace(0, 1e-8)).fillna(0.0)

    numeric_cols = [cname for cname in out.columns if cname != "time" and pd.api.types.is_numeric_dtype(out[cname])]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def get_extended_feature_columns() -> List[str]:
    """Stable feature list used by training and inference."""
    return [
        "atr",
        "atr_pct",
        "range",
        "body",
        "upper_wick",
        "lower_wick",
        "body_to_range",
        "close_open",
        "close_to_high",
        "close_to_low",
        "return_1",
        "return_3",
        "return_5",
        "return_10",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_diff_5_20",
        "trend_5",
        "trend_10",
        "volume_ma_5",
        "volume_ratio",
        "impulse",
        "balance",
        "impulse_norm",
        "cdi",
        "volatility_expansion",
        "rolling_range_5",
        "rolling_range_10",
        "rolling_range_20",
        "breakout_up",
        "breakout_down",
        "momentum_points",
        "price_position_20",
        "volatility_z",
    ]
