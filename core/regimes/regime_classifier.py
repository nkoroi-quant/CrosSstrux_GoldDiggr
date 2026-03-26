import pandas as pd


def classify_regime(features: pd.DataFrame) -> pd.DataFrame:
    """
    Simple but robust regime classifier.
    Guarantees 'regime' column even with minimal data.
    """
    df = features.copy()

    if "impulse" not in df.columns or "balance" not in df.columns:
        # Fallback if features are incomplete
        df["regime"] = "mid"
        return df

    impulse = df["impulse"].iloc[-1]
    balance = df["balance"].iloc[-1]
    cdi = df.get("cdi", pd.Series([0])).iloc[-1]

    if impulse == 1 and cdi > 0.25:
        regime = "high"
    elif balance == 1 and cdi < -0.25:
        regime = "low"
    elif abs(cdi) > 0.4:
        regime = "mid" if impulse == 1 else "low"
    else:
        regime = "mid"

    df["regime"] = regime
    return df