# tools/backtest_harness.py - Vectorized backtester reusing inference engine

import pandas as pd
from inference.engine import run_inference


def backtest(df: pd.DataFrame, asset: str):
    signals = []
    for i in range(20, len(df)):
        candles = df.iloc[i - 100 : i].to_dict("records")
        resp = run_inference(asset, "M1", candles, {})
        signals.append(resp)
    return pd.DataFrame(signals)
