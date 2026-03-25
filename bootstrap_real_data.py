"""
Normalize an exported broker CSV into CrossStrux raw parquet.

Expected input columns (any one of these variants per field):
- time / timestamp / Date
- open / Open
- high / High
- low / Low
- close / Close
- volume / tick_volume / real_volume
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from utils.parquet_compat import install as install_parquet_compat

install_parquet_compat()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    for src, dst in [
        ("timestamp", "time"),
        ("Timestamp", "time"),
        ("Date", "time"),
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
        ("Volume", "volume"),
    ]:
        if src in df.columns:
            rename_map[src] = dst

    df = df.rename(columns=rename_map)

    if "time" not in df.columns:
        raise ValueError("CSV must include a time/timestamp column")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Could not parse one or more time values")

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tick_volume" not in df.columns:
        if "volume" in df.columns:
            df["tick_volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        elif "real_volume" in df.columns:
            df["tick_volume"] = pd.to_numeric(df["real_volume"], errors="coerce").fillna(0)
        else:
            df["tick_volume"] = 0

    if "spread" not in df.columns:
        df["spread"] = 0

    if "real_volume" not in df.columns:
        df["real_volume"] = df["tick_volume"]

    return df.sort_values("time").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to broker CSV export")
    parser.add_argument("--asset", default="XAUUSD", help="Logical asset name")
    parser.add_argument("--output-dir", default=os.path.join("data", "raw"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = normalize_columns(df)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{args.asset}_M1.parquet")
    df.to_parquet(out, index=False)
    print(f"Saved {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
