
"""
Trade construction for CrossStrux v3.

The API returns the execution blueprint. GoldDiggr still computes the exact lot
size locally because that depends on the broker contract, account balance, and
user risk settings available inside MT5.
"""

from __future__ import annotations

from typing import Dict, Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rr_for_state(regime: str, market_state: str) -> float:
    regime = (regime or "mid").lower()
    market_state = (market_state or "TRANSITION").upper()

    base = {"low": 1.55, "mid": 2.00, "high": 2.40}.get(regime, 2.00)

    if market_state == "BREAKOUT":
        base += 0.20
    elif market_state == "TRENDING":
        base += 0.10
    elif market_state == "REVERSAL":
        base -= 0.10
    elif market_state == "RANGING":
        base -= 0.25

    return _clamp(base, 1.20, 3.00)


def _price_precision(price: float) -> int:
    if abs(price) >= 100:
        return 3
    return 5


def build_trade(
    signal: Dict[str, Any],
    latest_row,
    context: Dict[str, Any],
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a deterministic trade plan from the signal and structure.

    Uses ATR-aware buffers and structural levels.
    """
    action = str(signal["action"]).upper()
    price = float(latest_row["close"])
    atr = float(latest_row.get("atr", 0.0) or 0.0)
    atr_pct = float(latest_row.get("atr_pct", 0.0) or 0.0)

    key_high = float(structure["key_high"])
    key_low = float(structure["key_low"])
    midpoint = float(structure["midpoint"])

    regime = str(context.get("regime", "mid")).lower()
    market_state = str(context.get("market_state", "TRANSITION")).upper()
    volatility_state = str(context.get("volatility_state", "NORMAL")).upper()
    liquidity_state = str(context.get("liquidity_state", "UNCLEAR")).upper()
    session = str(context.get("session", "Unknown"))

    # XAUUSD likes wider stops than many FX pairs. Use ATR if present and add a
    # structural buffer when the market is volatile or sweeping liquidity.
    volatility_buffer = max(atr * 0.25, price * atr_pct * 0.30, 0.10)
    if volatility_state == "HIGH":
        volatility_buffer *= 1.15
    if liquidity_state == "SWEEPED":
        volatility_buffer *= 1.10

    rr = _rr_for_state(regime, market_state)

    if action == "BUY":
        entry = price
        structural_stop = min(key_low - volatility_buffer, entry - volatility_buffer)
        stop_loss = structural_stop
        risk_distance = max(entry - stop_loss, atr * 0.20, 0.10)
        take_profit = entry + risk_distance * rr
        bias = "BULLISH"
    else:
        entry = price
        structural_stop = max(key_high + volatility_buffer, entry + volatility_buffer)
        stop_loss = structural_stop
        risk_distance = max(stop_loss - entry, atr * 0.20, 0.10)
        take_profit = entry - risk_distance * rr
        bias = "BEARISH"

    # Confidence grows slightly with cleaner market context.
    confidence = float(context.get("probability", 0.0))
    confidence += 0.05 if market_state in {"TRENDING", "BREAKOUT"} else 0.02
    confidence += 0.03 if liquidity_state == "CLEAN" else 0.0
    confidence += 0.02 if str(signal.get("setup", "")).startswith("breakout") else 0.0
    confidence = min(0.99, confidence)

    # Profit shaping: scale out early, then trail.
    partial_tp = [
        {"rr": 1.0, "close_pct": 0.50},
        {"rr": 2.0, "close_pct": 0.25},
        {"rr": rr, "close_pct": 0.25},
    ]

    trail_after_rr = 1.5 if market_state != "RANGING" else 1.0
    breakeven_rr = 1.0
    max_hold_bars = 48 if market_state in {"TRENDING", "BREAKOUT"} else 24

    precision = _price_precision(price)

    return {
        "action": action,
        "entry": round(float(entry), precision),
        "stop_loss": round(float(stop_loss), precision),
        "take_profit": round(float(take_profit), precision),
        "rr": round(float(abs(take_profit - entry) / max(abs(entry - stop_loss), 1e-9)), 2),
        "confidence": round(float(confidence), 3),
        "bias": bias,
        "reason": signal["reason"],
        "setup": signal.get("setup"),
        "execution_type": "market",
        "market_state": market_state,
        "regime": regime,
        "volatility_state": volatility_state,
        "liquidity_state": liquidity_state,
        "session": session,
        "dynamic_threshold": float(signal.get("dynamic_threshold", context.get("dynamic_threshold", 0.65))),
        "persistence_score": float(signal.get("persistence_score", context.get("persistence_score", 0.0))),
        "trend_score": float(signal.get("trend_score", context.get("trend_score", 0.0))),
        "cooldown_bars": int(signal.get("cooldown_bars", context.get("cooldown_bars", 2))),
        "recommended_risk_pct": round(float(signal.get("recommended_risk_pct", context.get("recommended_risk_pct", 0.50))), 3),
        "lot_multiplier": round(float(context.get("risk_multiplier", 1.0)), 3),
        "partial_tp": partial_tp,
        "breakeven_at_rr": breakeven_rr,
        "trail_after_rr": trail_after_rr,
        "max_hold_bars": max_hold_bars,
        "news_blocked": bool(context.get("news_blocked", False)),
        "key_levels_used": {
            "key_high": round(key_high, precision),
            "key_low": round(key_low, precision),
            "midpoint": round(midpoint, precision),
        },
        "risk_multiplier": float(context.get("risk_multiplier", 1.0)),
        "signal_strength": round(float(signal.get("probability", context.get("probability", 0.0))), 3),
    }

