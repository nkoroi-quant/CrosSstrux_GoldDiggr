
"""
Response formatting for CrossStrux v3.

The response remains backward compatible with GoldDiggr v8/v9 fields while also
providing a structured v3 payload for richer downstream logic.
"""

from __future__ import annotations

from typing import Dict, Any


def build_response(
    *,
    asset: str,
    timeframe: str,
    session: Dict[str, Any],
    connection_status: str,
    context: Dict[str, Any],
    structure: Dict[str, Any],
    market_state: str,
    strategy_context: str,
    trade: Dict[str, Any] | None,
    spread_points: float | None,
    price: float | None,
) -> Dict[str, Any]:
    active_bias = trade["bias"] if trade else "NEUTRAL"
    last_signal = trade["action"] if trade else "NONE"
    signal_confidence = trade["confidence"] if trade else 0.0

    market = {
        "regime": context["regime"],
        "market_state": market_state,
        "volatility_state": context.get("volatility_state"),
        "liquidity_state": context.get("liquidity_state"),
        "trend_score": context.get("trend_score"),
        "persistence_score": context.get("persistence_score"),
        "session": session.get("session"),
        "session_timezone": session.get("timezone"),
    }

    signal = {
        "last_signal": last_signal,
        "active_bias": active_bias,
        "signal_confidence": signal_confidence,
        "probability": context["probability"],
        "transition_probability": context["transition_probability"],
        "cdi": context["cdi"],
        "severity": context["severity"],
        "confirmed_elevated": context["confirmed_elevated"],
        "consecutive_elevated": context["consecutive_elevated"],
        "dynamic_threshold": context.get("dynamic_threshold"),
        "setup": trade.get("setup") if trade else None,
        "reason": trade.get("reason") if trade else None,
        "strategy_context": strategy_context,
    }

    risk = {
        "risk_multiplier": context["risk_multiplier"],
        "recommended_risk_pct": context.get("recommended_risk_pct"),
        "lot_multiplier": context.get("risk_multiplier"),
        "spread_points": spread_points,
        "max_spread_points": context.get("max_spread_points"),
        "cooldown_bars": context.get("cooldown_bars"),
        "news_blocked": context.get("news_blocked", False),
        "drawdown_pct": context.get("drawdown_pct"),
        "persistence_required": context.get("persistence_required"),
        "max_positions": context.get("max_positions"),
        "exposure_blocked": context.get("exposure_blocked", False),
    }

    management = {
        "breakeven_at_rr": trade["breakeven_at_rr"] if trade else 1.0,
        "trail_after_rr": trade["trail_after_rr"] if trade else 1.5,
        "partial_tp": trade["partial_tp"] if trade else [{"rr": 1.0, "close_pct": 0.5}],
        "max_hold_bars": trade["max_hold_bars"] if trade else 24,
        "kill_switch": {
            "enabled": bool(context.get("kill_switch_enabled", True)),
            "daily_loss_limit_pct": context.get("daily_loss_limit_pct", 3.0),
        },
    }

    diagnostics = {
        "top_drift_feature": context.get("top_drift_feature"),
        "rolling_samples": context.get("rolling_samples", 0),
        "model_fallback": context.get("model_fallback"),
        "volatility_expansion": context.get("volatility_expansion"),
        "volatility_state": context.get("volatility_state"),
        "liquidity_state": context.get("liquidity_state"),
        "trend_score": context.get("trend_score"),
        "persistence_score": context.get("persistence_score"),
        "debug": {
            "impulse_up": bool(structure.get("impulse_up")),
            "impulse_down": bool(structure.get("impulse_down")),
            "breakout_up": bool(structure.get("breakout_up")),
            "breakout_down": bool(structure.get("breakout_down")),
            "liquidity_sweep_up": bool(structure.get("liquidity_sweep_up")),
            "liquidity_sweep_down": bool(structure.get("liquidity_sweep_down")),
            "bullish_order_block": bool(structure.get("bullish_order_block")),
            "bearish_order_block": bool(structure.get("bearish_order_block")),
        },
    }

    v3 = {
        "schema_version": "3.0",
        "meta": {
            "asset": asset,
            "symbol": asset,
            "timeframe": timeframe,
            "timestamp": session["eat_time"],
            "connection_status": connection_status,
            "generated_timezone": session.get("timezone", "EAT"),
        },
        "market": market,
        "signal": signal,
        "risk": risk,
        "trade": trade if trade else {"action": "NONE"},
        "management": management,
        "diagnostics": diagnostics,
    }

    response = {
        "schema_version": "3.0",
        "asset": asset,
        "symbol": asset,
        "timeframe": timeframe,
        "timestamp": session["eat_time"],
        "connection_status": connection_status,
        "session": session,
        "market_state": market_state,
        "regime": context["regime"],
        "probability": context["probability"],
        "transition_probability": context["transition_probability"],
        "cdi": context["cdi"],
        "severity": context["severity"],
        "confirmed_elevated": context["confirmed_elevated"],
        "consecutive_elevated": context["consecutive_elevated"],
        "risk_multiplier": context["risk_multiplier"],
        "top_drift_feature": context.get("top_drift_feature"),
        "rolling_samples": context.get("rolling_samples", 0),
        "model_fallback": context.get("model_fallback"),
        "last_signal": last_signal,
        "signal_confidence": signal_confidence,
        "strategy_context": strategy_context,
        "active_bias": active_bias,
        "spread_points": spread_points,
        "price": price,
        "dynamic_threshold": signal.get("dynamic_threshold"),
        "persistence_score": signal.get("persistence_score"),
        "trend_score": signal.get("trend_score"),
        "volatility_state": context.get("volatility_state"),
        "liquidity_state": context.get("liquidity_state"),
        "cooldown_bars": context.get("cooldown_bars"),
        "recommended_risk_pct": context.get("recommended_risk_pct"),
        "lot_multiplier": context.get("risk_multiplier"),
        "max_spread_points": context.get("max_spread_points"),
        "news_blocked": context.get("news_blocked", False),
        "persistence_required": context.get("persistence_required"),
        "max_positions": context.get("max_positions"),
        "drawdown_pct": context.get("drawdown_pct"),
        "breakeven_at_rr": management["breakeven_at_rr"],
        "trail_after_rr": management["trail_after_rr"],
        "max_hold_bars": management["max_hold_bars"],
        "key_levels": structure["key_levels"],
        "trade": trade if trade else {"action": "NONE"},
        "management": management,
        "debug": diagnostics["debug"],
        "market": market,
        "signal": signal,
        "risk": risk,
        "execution": trade if trade else {"action": "NONE"},
        "diagnostics": diagnostics,
        "v3": v3,
    }
    return response

