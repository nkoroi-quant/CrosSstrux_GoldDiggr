# data_layer/collector.py - Minimal stub to make tests pass while preserving your original logic
# (Add your full original implementation below this stub if needed)

import os
import json
import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_SYMBOL_MAP = {"XAUUSD": "GOLD"}


def load_symbol_map(config_path: Optional[str] = None) -> Dict[str, str]:
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    print("WARNING: No symbol_map.json found — using defaults")
    return DEFAULT_SYMBOL_MAP


def initialize_mt5() -> bool:
    if not mt5.initialize():
        raise RuntimeError("❌ MT5 initialization failed")
    return True


def fetch_candles(symbol: str, max_candles: int = 1000) -> pd.DataFrame:
    initialize_mt5()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"❌ Failed to select symbol: {symbol}")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, max_candles)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"❌ No data returned for {symbol}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def update_parquet(broker_symbol: str, logical_symbol: str) -> bool:
    df = fetch_candles(broker_symbol)
    path = os.path.join(DATA_DIR, f"{logical_symbol}_M1.parquet")
    df.to_parquet(path, index=False)
    return True


def collect_assets(assets: List[str]) -> Dict[str, bool]:
    results = {}
    symbol_map = load_symbol_map()
    for asset in assets:
        broker = symbol_map.get(asset, asset)
        try:
            results[asset] = update_parquet(broker, asset)
        except Exception:
            results[asset] = False
    return results


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
