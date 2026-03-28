from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adapter.decision_engine import decide_trade
from adapter.response_builder import build_response
from adapter.trade_builder import build_trade
from config.settings import settings
from core.features.feature_pipeline import build_features
from core.regimes.regime_classifier import classify_regime
from core.structure.balance import detect_balance
from core.structure.impulse import detect_impulse
from core.structure.levels import detect_key_levels, summarize_key_levels
from core.structure.sessions import get_session_context
from core.structure.volatility import volatility_expansion


def _safe_frame(candles: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = pd.DataFrame(candles).copy()
    if "time" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        else:
            raise ValueError("Each candle must include a time field")

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Each candle must include a {col} field")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time")
    return df.reset_index(drop=True)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = build_features(df)
    out = detect_impulse(out)
    out = detect_balance(out)
    out = volatility_expansion(out)
    out = detect_key_levels(out)
    return out


def _derive_bias(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 3:
        return "NEUTRAL"

    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)

    if closes.iloc[-1] > closes.iloc[-2] > closes.iloc[-3] and highs.iloc[-1] >= highs.iloc[-2] and lows.iloc[-1] >= lows.iloc[-2]:
        return "UP"
    if closes.iloc[-1] < closes.iloc[-2] < closes.iloc[-3] and highs.iloc[-1] <= highs.iloc[-2] and lows.iloc[-1] <= lows.iloc[-2]:
        return "DOWN"

    recent = closes.tail(min(5, len(closes)))
    if recent.iloc[-1] > recent.iloc[0]:
        return "UP"
    if recent.iloc[-1] < recent.iloc[0]:
        return "DOWN"
    return "NEUTRAL"


def _derive_sweep(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 6:
        return "NONE"

    latest = df.iloc[-1]
    previous = df.iloc[-6:-1]
    prev_high = float(previous["high"].max())
    prev_low = float(previous["low"].min())
    sweep_buffer = max(abs(float(latest["close"])) * 0.00005, 0.10)

    if float(latest["low"]) < (prev_low - sweep_buffer) and float(latest["close"]) > prev_low:
        return "BULL"
    if float(latest["high"]) > (prev_high + sweep_buffer) and float(latest["close"]) < prev_high:
        return "BEAR"
    return "NONE"


def _trend_score(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 6:
        return 0.0
    closes = df["close"].astype(float)
    atr = float(df.get("atr", pd.Series([0.0])).iloc[-1] or 0.0)
    if atr <= 0.0:
        atr = max(float(closes.iloc[-1]) * 0.001, 1e-8)
    move = float(closes.iloc[-1] - closes.iloc[-6])
    return float(np.tanh(move / (atr * 2.0 + 1e-8)))


def _persistence_score(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 4:
        return 0.0
    diffs = df["close"].astype(float).diff().tail(8).dropna()
    if diffs.empty:
        return 0.0
    dominant = 1.0 if diffs.mean() >= 0 else -1.0
    aligned = ((diffs > 0) & (dominant > 0)) | ((diffs < 0) & (dominant < 0))
    return float(aligned.mean())


def _volatility_state(vol_expansion: float) -> str:
    if vol_expansion >= 1.35:
        return "HIGH"
    if vol_expansion >= 1.10:
        return "EXPANDING"
    if vol_expansion <= 0.85:
        return "LOW"
    return "NORMAL"


def _liquidity_state(sweep: str, breakout_up: bool, breakout_down: bool) -> str:
    if sweep != "NONE":
        return "SWEEPED"
    if breakout_up or breakout_down:
        return "CLEAN"
    return "UNCLEAR"


def _market_state(h1_bias: str, m5_bias: str, sweep: str, cdi: float, trend_score: float) -> str:
    if sweep != "NONE" and abs(cdi) > 0.30:
        return "REVERSAL"
    if h1_bias == m5_bias and h1_bias != "NEUTRAL" and abs(trend_score) > 0.18:
        return "TRENDING"
    if abs(cdi) > 0.35:
        return "BREAKOUT"
    return "TRANSITION"


def _entry_precision_score(signal: str, m5_df: pd.DataFrame, m1_df: pd.DataFrame) -> float:
    if m5_df.empty or m1_df.empty or len(m1_df) < 3:
        return 0.0

    m5_bias = _derive_bias(m5_df)
    m5_sweep = _derive_sweep(m5_df)
    m1_bias = _derive_bias(m1_df)

    c0 = m1_df.iloc[-1]
    c1 = m1_df.iloc[-2]
    c2 = m1_df.iloc[-3]

    pt = max(float(c0["close"]) * 0.0001, 1e-8)
    body = abs(float(c0["close"]) - float(c0["open"])) / max(float(c0["high"]) - float(c0["low"]), pt)
    momentum = (float(c0["close"]) - float(c2["close"])) / pt

    bullish_engulf = (
        float(c0["close"]) > float(c0["open"])
        and float(c1["close"]) < float(c1["open"])
        and float(c0["close"]) > float(c1["open"])
        and float(c0["open"]) <= float(c1["close"])
    )
    bearish_engulf = (
        float(c0["close"]) < float(c0["open"])
        and float(c1["close"]) > float(c1["open"])
        and float(c0["close"]) < float(c1["open"])
        and float(c0["open"]) >= float(c1["close"])
    )
    micro_break_buy = float(c0["close"]) > float(c1["high"]) and float(c0["close"]) > float(c0["open"])
    micro_break_sell = float(c0["close"]) < float(c1["low"]) and float(c0["close"]) < float(c0["open"])

    score = 0.0
    if signal == "BUY":
        if bullish_engulf:
            score += 0.35
        if micro_break_buy:
            score += 0.25
        if float(c0["close"]) > float(c1["high"]):
            score += 0.10
        if m1_bias == "UP":
            score += 0.10
        if m5_bias == "UP":
            score += 0.15
        if m5_sweep == "BULL":
            score += 0.10
        if momentum > 4.0:
            score += 0.10
        score += min(body, 0.20)
    elif signal == "SELL":
        if bearish_engulf:
            score += 0.35
        if micro_break_sell:
            score += 0.25
        if float(c0["close"]) < float(c1["low"]):
            score += 0.10
        if m1_bias == "DOWN":
            score += 0.10
        if m5_bias == "DOWN":
            score += 0.15
        if m5_sweep == "BEAR":
            score += 0.10
        if momentum < -4.0:
            score += 0.10
        score += min(body, 0.20)

    return float(max(0.0, min(1.5, score)))


def run_inference(asset: str, timeframe: str, candles: Optional[List[Dict]] = None, request_context: Optional[Dict] = None) -> Dict:
    request_context = request_context or {}

    context_h1 = request_context.get("context_candles_h1") or request_context.get("h1_candles") or []
    context_m15 = request_context.get("context_candles_m15") or request_context.get("context_candles_m5") or request_context.get("candles") or candles or []
    entry_m1 = request_context.get("entry_candles_m1") or []
    entry_m5 = request_context.get("entry_candles_m5") or []

    h1_df = _safe_frame(context_h1) if context_h1 else pd.DataFrame()
    m15_df = _safe_frame(context_m15)
    if m15_df.empty:
        raise ValueError("CrossStrux requires at least one M15 candle for context analysis")

    m1_df = _safe_frame(entry_m1) if entry_m1 else pd.DataFrame()
    entry_m5_df = _safe_frame(entry_m5) if entry_m5 else m15_df.copy()

    h1_features = _prepare_features(h1_df) if not h1_df.empty else pd.DataFrame()
    m15_features = _prepare_features(m15_df)
    entry_m5_features = _prepare_features(entry_m5_df) if not entry_m5_df.empty else pd.DataFrame()
    entry_m1_features = _prepare_features(m1_df) if not m1_df.empty else pd.DataFrame()

    m15_latest = m15_features.iloc[-1]
    h1_latest = h1_features.iloc[-1] if not h1_features.empty else m15_latest
    entry_m5_latest = entry_m5_features.iloc[-1] if not entry_m5_features.empty else m15_latest
    entry_latest = entry_m1_features.iloc[-1] if not entry_m1_features.empty else None

    session = get_session_context()

    h1_bias = _derive_bias(h1_df) if not h1_df.empty else _derive_bias(m15_df)
    m15_bias = _derive_bias(m15_df)
    m5_bias = _derive_bias(entry_m5_df)
    m15_sweep = _derive_sweep(m15_df)
    m5_sweep = _derive_sweep(entry_m5_df)
    regime_df = classify_regime(m15_features.iloc[-1:])
    regime = str(regime_df["regime"].iloc[0])

    cdi = float(m15_latest.get("cdi", 0.0) or 0.0)
    volatility_expansion = float(m15_latest.get("volatility_expansion", 1.0) or 1.0)
    vol_state = _volatility_state(volatility_expansion)
    trend_score = _trend_score(m15_df)
    persistence_score = _persistence_score(m15_df)
    liquidity_state = _liquidity_state(
        m15_sweep,
        bool(int(m15_latest.get("breakout_up", 0) or 0)),
        bool(int(m15_latest.get("breakout_down", 0) or 0)),
    )

    market_state = _market_state(h1_bias, m15_bias, m15_sweep, cdi, trend_score)

    key_levels = summarize_key_levels(m15_latest)
    structure = {
        **key_levels,
        "key_levels": key_levels,
        "impulse_up": bool(int(entry_m5_latest.get("impulse", 0) or 0) and float(entry_m5_latest.get("impulse_dir", 0) or 0) > 0),
        "impulse_down": bool(int(entry_m5_latest.get("impulse", 0) or 0) and float(entry_m5_latest.get("impulse_dir", 0) or 0) < 0),
        "breakout_up": bool(int(entry_m5_latest.get("breakout_up", 0) or 0)),
        "breakout_down": bool(int(entry_m5_latest.get("breakout_down", 0) or 0)),
        "liquidity_sweep_up": bool(int(entry_m5_latest.get("liquidity_sweep_up", 0) or 0)),
        "liquidity_sweep_down": bool(int(entry_m5_latest.get("liquidity_sweep_down", 0) or 0)),
        "bullish_order_block": bool(int(entry_m5_latest.get("bullish_order_block", 0) or 0)),
        "bearish_order_block": bool(int(entry_m5_latest.get("bearish_order_block", 0) or 0)),
    }

    entry_precision_buy = _entry_precision_score("BUY", entry_m5_df, m1_df if not m1_df.empty else entry_m5_df.tail(3))
    entry_precision_sell = _entry_precision_score("SELL", entry_m5_df, m1_df if not m1_df.empty else entry_m5_df.tail(3))

    base_prob = 0.56 + min(0.10, abs(float(m15_latest.get("impulse_norm", 0.0) or 0.0)) * 0.18)
    if h1_bias == m15_bias and h1_bias != "NEUTRAL":
        base_prob += 0.05
    if m15_sweep != "NONE":
        base_prob += 0.03
    if market_state in {"TRENDING", "BREAKOUT"}:
        base_prob += 0.03
    if vol_state == "HIGH" and market_state == "BREAKOUT":
        base_prob += 0.02
    if vol_state == "LOW" and market_state == "TRANSITION":
        base_prob -= 0.02
    probability = float(max(0.52, min(0.84, base_prob)))
    transition_probability = float(max(0.05, min(0.95, 1.0 - probability + 0.08)))

    severity = 2 if abs(cdi) > 0.25 else 1
    confirmed_elevated = severity >= 2

    active_bias = "NEUTRAL"
    if h1_bias == "UP" and m15_bias == "UP":
        active_bias = "BULLISH"
    elif h1_bias == "DOWN" and m15_bias == "DOWN":
        active_bias = "BEARISH"
    elif m15_bias == "UP":
        active_bias = "BULLISH"
    elif m15_bias == "DOWN":
        active_bias = "BEARISH"

    context = {
        "regime": regime,
        "market_state": market_state,
        "volatility_state": vol_state,
        "liquidity_state": liquidity_state,
        "session": session.get("session"),
        "session_timezone": session.get("timezone"),
        "probability": probability,
        "transition_probability": transition_probability,
        "cdi": round(cdi, 3),
        "severity": severity,
        "confirmed_elevated": confirmed_elevated,
        "consecutive_elevated": 1 if confirmed_elevated else 0,
        "risk_multiplier": 0.55 if vol_state != "HIGH" else 0.50,
        "rolling_samples": int(len(m15_features)),
        "model_fallback": False,
        "volatility_expansion": round(volatility_expansion, 3),
        "trend_score": round(trend_score, 3),
        "persistence_score": round(persistence_score, 3),
        "top_drift_feature": "impulse_norm" if "impulse_norm" in m15_features.columns else None,
        "recommended_risk_pct": 0.50 if regime != "high" else 0.45,
        "cooldown_bars": 2 if market_state in {"TRENDING", "BREAKOUT"} else 3,
        "max_spread_points": request_context.get("max_spread_points", 80.0),
        "drawdown_pct": request_context.get("drawdown_pct", 0.0),
        "persistence_required": request_context.get("persistence_required", 2),
        "max_positions": request_context.get("max_positions", 3),
        "news_blocked": bool(request_context.get("news_block", False)),
        "kill_switch_enabled": True,
        "daily_loss_limit_pct": 3.0,
        "entry_precision_buy": round(entry_precision_buy, 3),
        "entry_precision_sell": round(entry_precision_sell, 3),
        "signal_confidence": probability,
        "active_bias": active_bias,
        "h1_bias": h1_bias,
        "m15_bias": m15_bias,
        "m15_sweep": m15_sweep,
        "m5_bias": m5_bias,
        "m5_sweep": m5_sweep,
    }

    signal = decide_trade(context, structure, request_context=request_context)
    trade = None
    if signal:
        trade = build_trade(signal, entry_m5_latest, context, structure)
        # Give the chosen direction a slight confidence lift when entry precision agrees.
        if trade["action"] == "BUY":
            trade["confidence"] = min(0.99, float(trade["confidence"]) + min(0.03, entry_precision_buy * 0.02))
        else:
            trade["confidence"] = min(0.99, float(trade["confidence"]) + min(0.03, entry_precision_sell * 0.02))

    price = float(entry_m5_latest["close"])
    response = build_response(
        asset=asset,
        timeframe=timeframe,
        session=session,
        connection_status="OK",
        context=context,
        structure=structure,
        market_state=market_state,
        strategy_context="H1_M15_CONTEXT__M5_M1_ENTRY",
        trade=trade,
        spread_points=request_context.get("spread_points"),
        price=price,
    )
    return response
