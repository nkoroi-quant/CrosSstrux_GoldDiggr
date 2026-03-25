
"""
Decision engine for CrossStrux v3.

This module translates model context + structure into an executable intent.
It stays conservative: the model must be strong, market quality must be good,
and the setup must survive persistence / spread / cooldown checks.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Iterable


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _signal_history_score(history: Iterable[str], expected: str) -> float:
    items = [str(x).upper() for x in history if str(x).upper() in {"BUY", "SELL", "NONE"}]
    if not items or expected not in {"BUY", "SELL"}:
        return 0.0

    recent = items[-5:]
    matches = sum(1 for item in recent if item == expected)
    return matches / max(len(recent), 1)


def decide_trade(
    context: Dict[str, Any],
    structure: Dict[str, Any],
    request_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return a signal dictionary or None when the setup is not strong enough.
    """
    request_context = request_context or {}

    regime = str(context.get("regime", "mid")).lower()
    market_state = str(context.get("market_state", "TRANSITION")).upper()
    volatility_state = str(context.get("volatility_state", "NORMAL")).upper()
    liquidity_state = str(context.get("liquidity_state", "UNCLEAR")).upper()
    session = str(context.get("session", "Unknown"))

    probability = _as_float(context.get("probability"))
    transition = _as_float(context.get("transition_probability"))
    cdi = _as_float(context.get("cdi"))
    severity = int(context.get("severity", 0) or 0)
    risk_multiplier = _as_float(context.get("risk_multiplier"), 1.0)
    persistence_score = _as_float(context.get("persistence_score"))
    trend_score = _as_float(context.get("trend_score"))
    spread_points = _as_float(request_context.get("spread_points"))
    max_spread_points = _as_float(request_context.get("max_spread_points", 80.0), 80.0)
    base_threshold = _as_float(request_context.get("confidence_threshold", 0.65), 0.65)
    drawdown_pct = _as_float(request_context.get("drawdown_pct"))
    losing_streak = int(request_context.get("losing_streak", 0) or 0)
    winning_streak = int(request_context.get("winning_streak", 0) or 0)
    bars_since_last_trade = int(request_context.get("bars_since_last_trade", 999) or 999)
    min_persistence = int(request_context.get("min_persistence", 2) or 2)
    signal_history = request_context.get("signal_history") or []
    news_block = bool(request_context.get("news_block", False))
    cooldown_active = bool(request_context.get("cooldown_active", False))

    if news_block or cooldown_active:
        return None

    if severity >= 3 or cdi >= 0.70:
        return None

    if transition >= 0.80:
        return None

    if spread_points > 0 and spread_points > max_spread_points:
        return None

    if probability < 0.10:
        return None

    volatility_expansion = _as_float(context.get("volatility_expansion"), 1.0)

    # Adaptive threshold: the model must clear a higher bar during volatility,
    # spread pressure, drawdown, and losing streaks.
    volatility_adj = _clamp((volatility_expansion - 1.0) * 0.06, -0.03, 0.12)
    spread_pressure = 0.0
    if spread_points > 0 and max_spread_points > 0:
        spread_ratio = spread_points / max_spread_points
        spread_pressure = _clamp((spread_ratio - 0.5) * 0.12, 0.0, 0.14)

    streak_pressure = _clamp((losing_streak - winning_streak) * 0.015, 0.0, 0.10)
    drawdown_pressure = _clamp(drawdown_pct * 0.25, 0.0, 0.12)
    transition_pressure = _clamp(transition * 0.08, 0.0, 0.10)
    cdi_pressure = _clamp(cdi * 0.08, 0.0, 0.10)

    session_bonus = 0.0
    if session in {"London", "New York"} and market_state in {"TRENDING", "BREAKOUT"}:
        session_bonus = -0.03
    elif session == "Asia" and market_state == "RANGING":
        session_bonus = 0.02
    elif session == "Off-hours":
        session_bonus = 0.03

    persistence_bonus = -0.04 if persistence_score >= 0.70 else 0.0
    signal_history_score = _signal_history_score(signal_history, "BUY") if context.get("active_bias") == "BULLISH" else _signal_history_score(signal_history, "SELL")
    history_bonus = -0.02 if signal_history_score >= 0.67 else 0.0

    dynamic_threshold = _clamp(
        base_threshold + volatility_adj + spread_pressure + streak_pressure + drawdown_pressure + transition_pressure + cdi_pressure + session_bonus + history_bonus + persistence_bonus,
        0.55,
        0.90,
    )

    if probability < dynamic_threshold:
        return None

    # Structural cues.
    impulse_up = bool(structure.get("impulse_up"))
    impulse_down = bool(structure.get("impulse_down"))
    breakout_up = bool(structure.get("breakout_up"))
    breakout_down = bool(structure.get("breakout_down"))
    liquidity_sweep_up = bool(structure.get("liquidity_sweep_up"))
    liquidity_sweep_down = bool(structure.get("liquidity_sweep_down"))
    bullish_order_block = bool(structure.get("bullish_order_block"))
    bearish_order_block = bool(structure.get("bearish_order_block"))
    key_levels = structure.get("key_levels", {})

    long_signal = False
    short_signal = False
    setup = None
    reason = None

    # Low regime: prefer sweep + reclaim.
    if regime == "low":
        if liquidity_sweep_down and bullish_order_block:
            long_signal = True
            setup = "sweep_reclaim_long"
            reason = "Low-regime sweep below support with bullish reclaim."
        elif liquidity_sweep_up and bearish_order_block:
            short_signal = True
            setup = "sweep_reclaim_short"
            reason = "Low-regime sweep above resistance with bearish reclaim."

    # Mid regime: continuation off impulse + order block.
    elif regime == "mid":
        if impulse_up and bullish_order_block:
            long_signal = True
            setup = "continuation_long"
            reason = "Mid-regime bullish continuation off impulse and order block."
        elif impulse_down and bearish_order_block:
            short_signal = True
            setup = "continuation_short"
            reason = "Mid-regime bearish continuation off impulse and order block."

    # High regime: breakout continuation.
    elif regime == "high":
        if breakout_up or (impulse_up and bullish_order_block):
            long_signal = True
            setup = "breakout_long"
            reason = "High-regime upside breakout continuation."
        elif breakout_down or (impulse_down and bearish_order_block):
            short_signal = True
            setup = "breakout_short"
            reason = "High-regime downside breakout continuation."

    # Fallbacks only when the broader context is strong.
    if not long_signal and not short_signal:
        active_bias = str(context.get("active_bias", "NEUTRAL")).upper()
        signal_confidence = _as_float(context.get("signal_confidence"), probability)
        if active_bias == "BULLISH" and signal_confidence >= dynamic_threshold and market_state in {"TRENDING", "BREAKOUT"}:
            long_signal = True
            setup = "bias_fallback_long"
            reason = "Strong bullish bias fallback with supportive market state."
        elif active_bias == "BEARISH" and signal_confidence >= dynamic_threshold and market_state in {"TRENDING", "BREAKOUT"}:
            short_signal = True
            setup = "bias_fallback_short"
            reason = "Strong bearish bias fallback with supportive market state."

    if not long_signal and not short_signal:
        return None

    action = "BUY" if long_signal else "SELL"

    # Confidence grows with model quality, persistence and structural alignment.
    structure_bonus = 0.0
    if market_state == "BREAKOUT":
        structure_bonus += 0.04
    elif market_state == "TRENDING":
        structure_bonus += 0.03
    elif market_state == "REVERSAL":
        structure_bonus += 0.02

    if liquidity_state == "CLEAN":
        structure_bonus += 0.02
    if trend_score >= 0.70:
        structure_bonus += 0.03
    if bars_since_last_trade > 3:
        structure_bonus += 0.01

    signal_confidence = _clamp(probability + structure_bonus - (transition * 0.10) - (cdi * 0.05), 0.0, 0.99)

    # Conservative regime-aware cooldown guidance.
    if regime == "low":
        cooldown_bars = 3
    elif regime == "mid":
        cooldown_bars = 2
    else:
        cooldown_bars = 1

    if market_state in {"RANGING", "TRANSITION"}:
        cooldown_bars += 1

    if persistence_score < (0.35 if min_persistence <= 2 else 0.45):
        return None

    # Base risk recommendation derived from regime and model quality.
    base_risk_pct = {"low": 0.35, "mid": 0.50, "high": 0.70}.get(regime, 0.50)
    signal_risk_adjustment = 0.15 * (signal_confidence - dynamic_threshold)
    recommended_risk_pct = _clamp(base_risk_pct * risk_multiplier + signal_risk_adjustment, 0.05, 1.25)

    return {
        "action": action,
        "reason": reason,
        "setup": setup,
        "regime": regime,
        "market_state": market_state,
        "volatility_state": volatility_state,
        "liquidity_state": liquidity_state,
        "session": session,
        "probability": probability,
        "signal_confidence": signal_confidence,
        "dynamic_threshold": dynamic_threshold,
        "persistence_score": persistence_score,
        "trend_score": trend_score,
        "risk_multiplier": risk_multiplier,
        "recommended_risk_pct": recommended_risk_pct,
        "cooldown_bars": cooldown_bars,
        "key_levels": key_levels,
        "thresholds": {
            "base": base_threshold,
            "dynamic": dynamic_threshold,
            "min_persistence": min_persistence,
        },
    }

