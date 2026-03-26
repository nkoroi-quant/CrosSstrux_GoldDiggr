import time
from typing import Dict, List
import pandas as pd
from core.features.feature_pipeline import build_features
from core.regimes.regime_classifier import classify_regime
from inference.loader import load_asset_bundle
from adapter.response_builder import build_response
from config.settings import settings

def run_inference(asset: str, timeframe: str, candles: List[Dict], request_context: Dict) -> Dict:
    df_new = pd.DataFrame(candles)
    df_new["time"] = pd.to_datetime(df_new["time"])
    features = build_features(df_new)

    regime_df = classify_regime(features.iloc[-1:])
    regime = str(regime_df["regime"].iloc[0])

    prob_raw = 0.58 + (features["impulse_norm"].iloc[-1] * 0.22)
    prob_raw = min(0.79, max(0.53, prob_raw))

    cdi = float(features["cdi"].iloc[-1]) if "cdi" in features.columns else 0.0
    severity = 2 if abs(cdi) > 0.25 else 1

    adj_prob = prob_raw * (1.18 if severity >= 2 else 1.0)   # strong elevated boost
    if abs(features["close"].diff().iloc[-5:].mean()) > 40:   # momentum penalty
        adj_prob *= 0.92

    adj_prob = min(0.82, adj_prob)

    response = build_response(
        asset=asset,
        timeframe=timeframe,
        session=request_context.get("session", {}),
        connection_status="OK",
        context={
            "regime": regime,
            "probability": float(prob_raw),
            "cdi": round(cdi, 3),
            "severity": severity,
            "confirmed_elevated": severity >= 2,
            "risk_multiplier": 0.55,
            "rolling_samples": len(features),
        },
        structure={},
        market_state="BREAKOUT" if abs(cdi) > 0.3 else "RANGING",
        strategy_context="Active",
        trade=None,
        spread_points=request_context.get("spread_points"),
    )
    return response