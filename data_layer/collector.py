# data_layer/collector.py - MT5 data loader with graceful fallback for test/dev environments.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils.parquet_compat import install as install_parquet_compat

install_parquet_compat()

try:  # pragma: no cover - optional runtime dependency
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - makes tests/local development possible
    mt5 = None

DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_SYMBOL_MAP = {"XAUUSD": "GOLD"}


def load_symbol_map(config_path: Optional[str] = None) -> Dict[str, str]:
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    print("WARNING: No symbol_map.json found — using defaults")
    return DEFAULT_SYMBOL_MAP.copy()


def initialize_mt5() -> bool:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed")
    if not mt5.initialize():
        raise RuntimeError("❌ MT5 initialization failed")
    return True


def fetch_candles(symbol: str, max_candles: int = 1000) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed")

    initialize_mt5()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"❌ Failed to select symbol: {symbol}")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, max_candles)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"❌ No data returned for {symbol}")

    df = pd.DataFrame(rates)
    if "time" not in df.columns:
        raise RuntimeError("MT5 rates payload missing time column")
    df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    cols = [c for c in ["time", "open", "high", "low", "close", "tick_volume"] if c in df.columns]
    return df[cols].copy()


def update_parquet(broker_symbol: str, logical_symbol: str) -> bool:
    df = fetch_candles(broker_symbol)
    path = os.path.join(DATA_DIR, f"{logical_symbol}_M1.parquet")
    df.to_parquet(path, index=False)
    return True


def collect_assets(assets: List[str]) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    symbol_map = load_symbol_map()
    for asset in assets:
        broker = symbol_map.get(asset, asset)
        try:
            results[asset] = update_parquet(broker, asset)
        except Exception:
            results[asset] = False
    return results


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return df.copy()
    return df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
