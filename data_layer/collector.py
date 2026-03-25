# data_layer/collector.py

"""
CrossStrux Data Collector (FINAL)

- MT5 ingestion (M1)
- Strict parquet output (pyarrow)
- Broker-safe symbol handling
- No silent failures
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Optional

import pandas as pd
import MetaTrader5 as mt5

from utils.parquet_compat import write_parquet

# ================= CONFIG ================= #
DATA_DIR = "data/raw"
MAX_CANDLES = 90000
CONFIG_PATH = "config/symbol_map.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= SYMBOL MAP ================= #
def load_symbol_map(config_path: str = CONFIG_PATH) -> Dict[str, str]:
    if not os.path.exists(config_path):
        logger.warning("No symbol_map.json found — using defaults")
        return {
            "XAUUSD": "GOLD"
        }

    with open(config_path, "r") as f:
        return json.load(f)

# ================= MT5 ================= #
def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError("❌ MT5 initialization failed")

    logger.info("✅ MT5 initialized")


def shutdown_mt5():
    mt5.shutdown()
    logger.info("MT5 shutdown")


def ensure_symbol(symbol: str):
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"❌ Failed to select symbol: {symbol}")

# ================= FETCH ================= #
def fetch_candles(symbol: str, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"❌ No data returned for {symbol}")

    df = pd.DataFrame(rates)

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Normalize volume column
    if "tick_volume" not in df.columns:
        if "volume" in df.columns:
            df["tick_volume"] = df["volume"]
        elif "real_volume" in df.columns:
            df["tick_volume"] = df["real_volume"]
        else:
            df["tick_volume"] = 0

    # Ensure required columns
    required = ["time", "open", "high", "low", "close", "tick_volume"]
    df = df[required]

    return df

# ================= UPDATE ================= #
def update_parquet(logical: str, broker: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{logical}_M1.parquet")

    logger.info(f"Updating {logical} (broker: {broker})")

    ensure_symbol(broker)

    df_new = fetch_candles(broker, MAX_CANDLES)
    logger.info(f"Fetched {len(df_new)} candles")

    # Merge with existing
    if os.path.exists(path):
        try:
            df_old = pd.read_parquet(path, engine="pyarrow")
            df = pd.concat([df_old, df_new])
            df = df.drop_duplicates(subset=["time"], keep="last")
        except Exception:
            logger.warning("Existing parquet unreadable — replacing")
            df = df_new
    else:
        df = df_new

    df = df.sort_values("time").tail(MAX_CANDLES).reset_index(drop=True)

    write_parquet(df, path)

    logger.info(f"✅ Saved {logical}: {len(df)} rows → {path}")
    return True

# ================= MAIN ================= #
def collect_assets(assets: List[str]):
    symbol_map = load_symbol_map()

    results = {}

    for asset in assets:
        if asset not in symbol_map:
            logger.error(f"{asset} not in symbol map")
            results[asset] = False
            continue

        try:
            broker = symbol_map[asset]
            update_parquet(asset, broker)
            results[asset] = True
        except Exception as e:
            logger.error(f"{asset} failed: {e}")
            results[asset] = False

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=str, default="XAUUSD")

    args = parser.parse_args()
    assets = [a.strip() for a in args.assets.split(",")]

    logger.info(f"Starting collection: {assets}")

    initialize_mt5()

    try:
        results = collect_assets(assets)
        success = sum(results.values())
        logger.info(f"Done: {success}/{len(results)} successful")
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()