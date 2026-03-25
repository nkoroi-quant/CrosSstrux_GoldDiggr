"""
CrossStrux v2 EA simulator.

This sends the latest candles to the server and prints the response that
GoldDiggr would consume inside MT5.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data/raw"
DEFAULT_SERVER = "http://127.0.0.1:8000/analyze"
DEFAULT_N_CANDLES = 200


def load_candles(asset: str, n: int):
    path = os.path.join(DATA_DIR, f"{asset}_M1.parquet")
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        return None

    df = pd.read_parquet(path)
    df = df.tail(n).reset_index(drop=True)
    return df


def build_payload(asset: str, df: pd.DataFrame) -> Dict:
    candles = []
    for _, row in df.iterrows():
        candle = {
            "time": str(row["time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": (
                float(row["tick_volume"])
                if "tick_volume" in df.columns
                else float(row.get("volume", 0))
            ),
            "spread": float(row["spread"]) if "spread" in df.columns else 0.0,
        }
        candles.append(candle)

    return {
        "asset": asset,
        "timeframe": "M1",
        "spread_points": (
            float(df["spread"].iloc[-1]) if "spread" in df.columns and len(df) else 0.0
        ),
        "bid": float(df["close"].iloc[-1]) if len(df) else None,
        "ask": float(df["close"].iloc[-1]) if len(df) else None,
        "candles": candles,
    }


def send_request(server_url: str, payload: Dict):
    try:
        logger.info(f"POST {server_url}")
        logger.info(f"Payload size: {len(json.dumps(payload))} bytes")
        r = requests.post(server_url, json=payload, timeout=30)
        logger.info(f"Response status: {r.status_code}")
        if r.status_code == 200:
            return r.json()
        logger.error(r.text)
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Server not reachable")
        return None


def print_response(result: Dict):
    print("\n" + "=" * 80)
    print("CROSSSTRUX RESPONSE")
    print("=" * 80)
    for key in [
        "asset",
        "regime",
        "probability",
        "transition_probability",
        "cdi",
        "severity",
        "risk_multiplier",
        "market_state",
        "last_signal",
        "signal_confidence",
        "active_bias",
        "strategy_context",
    ]:
        if key in result:
            print(f"{key}: {result.get(key)}")

    if result.get("key_levels"):
        print("key_levels:", result["key_levels"])
    if result.get("trade"):
        print("trade:", result["trade"])
    print("=" * 80)


def simulate_asset(asset: str, server: str, n: int):
    df = load_candles(asset, n)
    if df is None or df.empty:
        return False

    payload = build_payload(asset, df)
    result = send_request(server, payload)
    if result is None:
        return False

    print_response(result)
    return True


def main():
    global DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="XAUUSD")
    parser.add_argument("--n", type=int, default=DEFAULT_N_CANDLES)
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    DATA_DIR = args.data_dir

    ok = simulate_asset(args.asset, args.server, args.n)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
