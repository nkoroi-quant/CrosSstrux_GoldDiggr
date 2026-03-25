"""
Generate synthetic XAUUSD M1 data for smoke testing the v2 pipeline.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from utils.parquet_compat import install as install_parquet_compat

install_parquet_compat()


DATA_DIR = os.path.join("data", "raw")
OUTPUT = os.path.join(DATA_DIR, "XAUUSD_M1.parquet")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    np.random.seed(42)
    n = 1000
    base = 2340.0
    drift = np.cumsum(np.random.normal(0.0, 0.65, n))
    close = base + drift
    open_ = np.roll(close, 1)
    open_[0] = base
    high = np.maximum(open_, close) + np.abs(np.random.normal(0.0, 0.35, n))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0.0, 0.35, n))
    volume = np.random.randint(100, 2000, n)
    spread = np.random.randint(18, 42, n)
    times = pd.date_range(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="min",
        tz="UTC",
    )

    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": volume,
        "spread": spread,
        "real_volume": volume,
    })
    df.to_parquet(OUTPUT, index=False)
    print(f"Saved synthetic XAUUSD data to {OUTPUT}")


if __name__ == "__main__":
    main()
