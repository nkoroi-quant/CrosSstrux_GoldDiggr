
# inference/engine.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.features.feature_pipeline import build_features, validate_input_columns
from core.regimes.regime_classifier import classify_regime
from core.regimes.stability import regime_stability
from core.structure.impulse import detect_impulse
from core.structure.balance import detect_balance
from core.structure.volatility import volatility_expansion
from core.structure.levels import detect_key_levels, summarize_key_levels
from core.structure.sessions import get_session_context
from adapter.decision_engine import decide_trade
from adapter.trade_builder import build_trade
from adapter.response_builder import build_response
from inference.loader import load_asset_bundle


# ================= CONSTANTS ================= #
MIN_CANDLES = 20
MIN_DRIFT_SAMPLES = 50

GOVERNANCE_THRESHOLDS = {
    "low": [0.10, 0.25, 0.50],
    "mid": [0.12, 0.28, 0.55],
    "high": [0.15, 0.35, 0.60],
}

RISK_MULTIPLIER = {
    0: 1.00,
    1: 0.80,
    2: 0.55,
    3: 0.00,
}

FALLBACK_CHAIN = {
    "low": ["mid", "high"],
    "mid": ["high", "low"],
    "high": ["mid", "low"],
}

VOLATILITY_STATE = {
    "low": (0.0, 0.90),
    "normal": (0.90, 1.25),
    "high": (1.25, float("inf")),
}


# ================= REQUIRED FUNCTIONS ================= #

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _series_numeric(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    return arr.dropna().to_numpy(dtype=float)


def compute_psi(expected: List[float], actual: List[float], buckets: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) < MIN_DRIFT_SAMPLES or len(actual) < MIN_DRIFT_SAMPLES:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    breaks = np.unique(np.quantile(expected, quantiles))

    if len(breaks) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breaks)
    actual_counts, _ = np.histogram(actual, bins=breaks)

    expected_perc = expected_counts / max(expected_counts.sum(), 1)
    actual_perc = actual_counts / max(actual_counts.sum(), 1)

    eps = 1e-6
    expected_perc = np.where(expected_perc == 0, eps, expected_perc)
    actual_perc = np.where(actual_perc == 0, eps, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(max(0.0, psi))


def select_regime(asset: str, impulse_norm: float, metadata: Optional[Dict[str, Any]] = None) -> str:
    metadata = metadata or {}
    low_th = float(metadata.get("low_threshold", 0.35))
    high_th = float(metadata.get("high_threshold", 0.85))

    if impulse_norm <= low_th:
        return "low"
    if impulse_norm <= high_th:
        return "mid"
    return "high"


def classify_severity(regime: str, cdi: float) -> int:
    t1, t2, t3 = GOVERNANCE_THRESHOLDS.get(regime, GOVERNANCE_THRESHOLDS["mid"])

    if cdi < t1:
        return 0
    if cdi < t2:
        return 1
    if cdi < t3:
        return 2
    return 3


def get_model_with_fallback(bundle, regime: str):
    if regime in bundle.models:
        return bundle.models[regime], regime

    for fallback in FALLBACK_CHAIN.get(regime, []):
        if fallback in bundle.models:
            return bundle.models[fallback], fallback

    raise ValueError("No model available")


# ================= MODEL / STRUCTURE HELPERS ================= #

def _estimate_transition_probability(bundle, features_df: pd.DataFrame, regime: str) -> Tuple[float, Optional[str]]:
    transition_model = getattr(bundle, "transition_model", None)
    transition_features = list(bundle.metadata.get("transition_features", []))
    if transition_model is not None and transition_features:
        try:
            row = features_df.iloc[-1:].copy()
            for col in transition_features:
                if col == "regime_encoded":
                    row[col] = {"low": 0, "mid": 1, "high": 2}.get(regime, 1)
                elif col not in row.columns:
                    row[col] = 0.0
            X_t = row[transition_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            prob = float(transition_model.predict_proba(X_t)[0][1])
            return _clamp(prob, 0.0, 0.99), "transition_model"
        except Exception:
            pass

    latest = features_df.iloc[-1]
    volatility = _safe_float(latest.get("volatility_expansion"), 1.0)
    balance = _safe_float(latest.get("balance"), 0.0)
    regime_stab = _safe_float(latest.get("regime_stability"), 0.0)
    impulse_norm = _safe_float(latest.get("impulse_norm"), 0.0)

    estimate = 0.15
    estimate += _clamp((volatility - 1.0) * 0.18, 0.0, 0.25)
    estimate += _clamp((1.0 - regime_stab) * 0.22, 0.0, 0.22)
    estimate += _clamp(balance * 0.10, 0.0, 0.10)
    estimate += _clamp(impulse_norm * 0.04, 0.0, 0.15)
    if regime == "high":
        estimate += 0.05
    elif regime == "low":
        estimate += 0.02
    return _clamp(estimate, 0.0, 0.95), None


def _compute_drift(bundle, features_df: pd.DataFrame) -> Tuple[float, Optional[str]]:
    baselines = getattr(bundle, "baseline", {}) or {}
    latest_window = features_df.tail(120)
    if latest_window.empty:
        return 0.0, None

    best_psi = 0.0
    best_feature = None

    for regime_name, baseline in baselines.items():
        if not isinstance(baseline, dict):
            continue
        for feature_name, expected_values in baseline.items():
            if feature_name not in latest_window.columns:
                continue
            expected = np.asarray(expected_values, dtype=float)
            actual = _series_numeric(latest_window[feature_name])
            psi = compute_psi(expected.tolist(), actual.tolist())
            if psi > best_psi:
                best_psi = psi
                best_feature = feature_name

    return float(best_psi), best_feature


def _volatility_state(volatility_expansion: float) -> str:
    if volatility_expansion < VOLATILITY_STATE["low"][1]:
        return "LOW"
    if volatility_expansion < VOLATILITY_STATE["normal"][1]:
        return "NORMAL"
    return "HIGH"


def _market_state(latest_row: pd.Series) -> str:
    impulse_up = bool(_safe_int(latest_row.get("impulse_dir", 0)) > 0)
    impulse_down = bool(_safe_int(latest_row.get("impulse_dir", 0)) < 0)
    breakout_up = bool(_safe_int(latest_row.get("breakout_up", 0)))
    breakout_down = bool(_safe_int(latest_row.get("breakout_down", 0)))
    sweep_up = bool(_safe_int(latest_row.get("liquidity_sweep_up", 0)))
    sweep_down = bool(_safe_int(latest_row.get("liquidity_sweep_down", 0)))
    bullish_ob = bool(_safe_int(latest_row.get("bullish_order_block", 0)))
    bearish_ob = bool(_safe_int(latest_row.get("bearish_order_block", 0)))
    balance = _safe_float(latest_row.get("balance"), 0.0)
    volatility = _safe_float(latest_row.get("volatility_expansion"), 1.0)

    if breakout_up or breakout_down:
        return "BREAKOUT"
    if (sweep_down and bullish_ob) or (sweep_up and bearish_ob):
        return "REVERSAL"
    if (impulse_up or impulse_down) and volatility > 1.15 and balance < 0.70:
        return "TRENDING"
    if balance >= 0.60 and volatility < 1.05:
        return "RANGING"
    return "TRANSITION"


def _liquidity_state(latest_row: pd.Series) -> str:
    sweep_up = bool(_safe_int(latest_row.get("liquidity_sweep_up", 0)))
    sweep_down = bool(_safe_int(latest_row.get("liquidity_sweep_down", 0)))
    if sweep_up or sweep_down:
        return "SWEEPED"
    if bool(_safe_int(latest_row.get("bullish_order_block", 0))) or bool(_safe_int(latest_row.get("bearish_order_block", 0))):
        return "CLEAN"
    return "UNCLEAR"


def _trend_score(latest_row: pd.Series) -> float:
    impulse_norm = _safe_float(latest_row.get("impulse_norm"), 0.0)
    volatility = _safe_float(latest_row.get("volatility_expansion"), 1.0)
    score = (impulse_norm / 2.5) * 0.7 + max(0.0, volatility - 0.95) * 0.3
    return _clamp(score, 0.0, 0.99)


def _persistence_score(df: pd.DataFrame, latest_row: pd.Series, market_state: str) -> float:
    recent = df.tail(4)
    if recent.empty:
        return 0.0

    direction = 1 if market_state in {"BREAKOUT", "TRENDING"} and _safe_int(latest_row.get("impulse_dir", 0)) >= 0 else -1 if market_state in {"REVERSAL"} or _safe_int(latest_row.get("impulse_dir", 0)) < 0 else 0
    if direction == 0:
        return 0.45 * _safe_float(latest_row.get("regime_stability"), 0.0)

    hits = 0
    for _, row in recent.iterrows():
        candle_dir = 1 if _safe_float(row.get("close"), 0.0) >= _safe_float(row.get("open"), 0.0) else -1
        impulse_dir = _safe_int(row.get("impulse_dir", 0))
        if direction > 0 and candle_dir > 0 and impulse_dir >= 0:
            hits += 1
        elif direction < 0 and candle_dir < 0 and impulse_dir <= 0:
            hits += 1

    alignment = hits / len(recent)
    stability = _safe_float(latest_row.get("regime_stability"), 0.0)
    score = alignment * 0.65 + stability * 0.35
    return _clamp(score, 0.0, 0.99)


def _session_multiplier(session_label: str, market_state: str) -> float:
    session_label = session_label or "Unknown"
    if session_label in {"London", "New York"}:
        return 1.05 if market_state in {"TRENDING", "BREAKOUT"} else 1.0
    if session_label == "Asia":
        return 0.90 if market_state == "RANGING" else 0.95
    return 0.85


def _cdi_from_components(psi: float, transition_probability: float, volatility_expansion: float, regime_stability_value: float) -> float:
    psi_component = _clamp(psi / 3.0, 0.0, 1.0)
    volatility_component = _clamp((volatility_expansion - 1.0) / 1.5, 0.0, 1.0)
    stability_component = 1.0 - _clamp(regime_stability_value, 0.0, 1.0)
    cdi = (psi_component * 0.40) + (transition_probability * 0.30) + (volatility_component * 0.20) + (stability_component * 0.10)
    return _clamp(cdi, 0.0, 0.99)


# ================= MAIN ================= #

def run_inference(
    asset: str,
    timeframe: str,
    candles: List[Dict[str, Any]],
    request_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    request_context = request_context or {}

    if len(candles) < MIN_CANDLES:
        raise ValueError("Insufficient candles (need at least 20)")

    bundle = load_asset_bundle(asset)

    df = pd.DataFrame(candles)
    df = validate_input_columns(df)

    # STRUCTURE
    df = detect_impulse(df)
    df = detect_balance(df)
    df = volatility_expansion(df)
    df = classify_regime(df)
    df = regime_stability(df)
    df = detect_key_levels(df)

    # FEATURES
    features_df = build_features(df)

    if features_df.empty:
        raise ValueError("Feature pipeline returned empty dataframe")

    latest_time = features_df.iloc[-1]["time"]
    if "time" in df.columns and (df["time"] == latest_time).any():
        latest_row = df[df["time"] == latest_time].iloc[-1]
    else:
        latest_row = df.iloc[-1]

    if "close" not in latest_row or latest_row["close"] is None:
        raise ValueError("Missing close price")

    feature_cols = bundle.metadata["feature_columns"]

    X = features_df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()

    if X.empty:
        raise ValueError("Invalid model input")

    X_single = X.iloc[-1:].values

    impulse_norm = _safe_float(features_df.iloc[-1].get("impulse_norm"), 0.0)
    regime = select_regime(asset, impulse_norm, bundle.metadata)

    model, fallback = get_model_with_fallback(bundle, regime)
    probability = float(model.predict_proba(X_single)[0][1])

    transition_probability, transition_source = _estimate_transition_probability(bundle, features_df, regime)
    drift_psi, top_drift_feature = _compute_drift(bundle, features_df)

    volatility_expansion_value = _safe_float(latest_row.get("volatility_expansion"), 1.0)
    regime_stability_value = _safe_float(latest_row.get("regime_stability"), 0.0)
    market_state = _market_state(latest_row)
    volatility_state = _volatility_state(volatility_expansion_value)
    liquidity_state = _liquidity_state(latest_row)
    trend_score = _trend_score(latest_row)
    persistence_score = _persistence_score(df, latest_row, market_state)
    session = get_session_context()

    cdi = _cdi_from_components(drift_psi, transition_probability, volatility_expansion_value, regime_stability_value)
    severity = classify_severity(regime, cdi)
    risk_multiplier = _clamp(RISK_MULTIPLIER.get(severity, 0.0) * _session_multiplier(session["session"], market_state), 0.0, 1.25)
    consecutive_elevated = severity if severity > 0 else 0
    confirmed_elevated = bool(severity >= 2 and persistence_score >= 0.55)

    structure = {
        "impulse_up": bool(_safe_float(latest_row.get("impulse_dir", 0), 0.0) > 0),
        "impulse_down": bool(_safe_float(latest_row.get("impulse_dir", 0), 0.0) < 0),
        "breakout_up": bool(_safe_int(latest_row.get("breakout_up", 0))),
        "breakout_down": bool(_safe_int(latest_row.get("breakout_down", 0))),
        "liquidity_sweep_up": bool(_safe_int(latest_row.get("liquidity_sweep_up", 0))),
        "liquidity_sweep_down": bool(_safe_int(latest_row.get("liquidity_sweep_down", 0))),
        "bullish_order_block": bool(_safe_int(latest_row.get("bullish_order_block", 0))),
        "bearish_order_block": bool(_safe_int(latest_row.get("bearish_order_block", 0))),
        "key_levels": summarize_key_levels(latest_row),
    }

    base_dynamic_threshold = _clamp(
        _safe_float(request_context.get("confidence_threshold", 0.65), 0.65)
        + (0.03 if volatility_state == "HIGH" else 0.0)
        + (0.02 if market_state == "TRANSITION" else 0.0)
        + (0.03 if severity >= 2 else 0.0),
        0.50,
        0.85,
    )

    request_context = {
        **request_context,
        "confidence_threshold": base_dynamic_threshold,
        "dynamic_threshold": base_dynamic_threshold,
        "volatility_expansion": volatility_expansion_value,
        "max_spread_points": _safe_float(request_context.get("max_spread_points", 80.0), 80.0),
        "signal_history": request_context.get("signal_history") or [],
        "min_persistence": _safe_int(request_context.get("min_persistence", 2), 2),
        "news_block": bool(request_context.get("news_block", False)),
        "cooldown_active": bool(request_context.get("cooldown_active", False)),
        "drawdown_pct": _safe_float(request_context.get("drawdown_pct", 0.0), 0.0),
        "losing_streak": _safe_int(request_context.get("losing_streak", 0), 0),
        "winning_streak": _safe_int(request_context.get("winning_streak", 0), 0),
        "bars_since_last_trade": _safe_int(request_context.get("bars_since_last_trade", 999), 999),
        "max_positions": _safe_int(request_context.get("max_positions", 1), 1),
        "exposure_blocked": bool(request_context.get("exposure_blocked", False)),
    }

    context = {
        "regime": regime,
        "probability": probability,
        "transition_probability": transition_probability,
        "transition_source": transition_source,
        "cdi": cdi,
        "severity": severity,
        "confirmed_elevated": confirmed_elevated,
        "consecutive_elevated": consecutive_elevated,
        "risk_multiplier": risk_multiplier,
        "top_drift_feature": top_drift_feature,
        "rolling_samples": len(features_df),
        "model_fallback": fallback if fallback != regime else None,
        "market_state": market_state,
        "volatility_state": volatility_state,
        "volatility_expansion": volatility_expansion_value,
        "liquidity_state": liquidity_state,
        "trend_score": trend_score,
        "persistence_score": persistence_score,
        "session": session["session"],
        "recommended_risk_pct": None,
        "max_spread_points": request_context["max_spread_points"],
        "cooldown_bars": 2,
        "news_blocked": request_context.get("news_block", False),
        "drawdown_pct": request_context.get("drawdown_pct", 0.0),
        "persistence_required": request_context.get("min_persistence", 2),
        "max_positions": request_context.get("max_positions", 1),
        "exposure_blocked": request_context.get("exposure_blocked", False),
        "daily_loss_limit_pct": request_context.get("daily_loss_limit_pct", 3.0),
        "kill_switch_enabled": True,
    }

    signal = decide_trade(context, structure, request_context=request_context)
    trade = build_trade(signal, latest_row, context, structure) if signal else None
    if trade is not None:
        context["recommended_risk_pct"] = trade.get("recommended_risk_pct")
        context["cooldown_bars"] = trade.get("cooldown_bars", context["cooldown_bars"])

    price = request_context.get("price") or latest_row.get("close")
    price = _safe_float(price, _safe_float(latest_row.get("close"), 0.0))
    spread = request_context.get("spread_points")
    spread = float(spread) if spread is not None else None

    strategy_context = f"{market_state} | {regime.upper()} | {volatility_state} | {liquidity_state}"
    if signal is None:
        strategy_context = f"{strategy_context} | HOLD"
    else:
        strategy_context = f"{strategy_context} | {signal.get('setup', 'setup')}"

    return build_response(
        asset=asset,
        timeframe=timeframe,
        session=session,
        connection_status="OK",
        context=context,
        structure=structure,
        market_state=market_state,
        strategy_context=strategy_context,
        trade=trade,
        spread_points=spread,
        price=price,
    )

