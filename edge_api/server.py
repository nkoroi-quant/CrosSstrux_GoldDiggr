
"""
CrossStrux v3 inference server.

This keeps backward compatibility with the original /predict contract while
exposing the richer GoldDiggr-compatible payload through /analyze.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference.engine import (
    compute_psi,
    select_regime as _select_regime,
    classify_severity,
    get_model_with_fallback,
    FALLBACK_CHAIN,
    GOVERNANCE_THRESHOLDS,
    RISK_MULTIPLIER,
    MIN_DRIFT_SAMPLES,
    run_inference,
)
from inference.loader import loaded_assets, load_asset_bundle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrossStrux v3 API", version="3.0")

# Backward-compatible globals used by the original tests and scripts.
asset_metadata: Dict[str, Dict[str, Any]] = {}
asset_models: Dict[str, Any] = {}
asset_transition_models: Dict[str, Any] = {}
asset_state: Dict[str, Any] = {}


class Candle(BaseModel):
    timestamp: Optional[str] = None
    time: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    tick_volume: Optional[float] = None
    spread: Optional[float] = None


class PredictRequest(BaseModel):
    asset: str
    timeframe: str = "M1"
    candles: List[Candle]
    spread_points: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    confidence_threshold: Optional[float] = None
    max_spread_points: Optional[float] = None

    # Optimisation inputs from GoldDiggr / simulation.
    drawdown_pct: Optional[float] = None
    daily_loss_pct: Optional[float] = None
    winning_streak: Optional[int] = None
    losing_streak: Optional[int] = None
    bars_since_last_trade: Optional[int] = None
    open_positions: Optional[int] = None
    max_positions: Optional[int] = None
    min_persistence: Optional[int] = None
    cooldown_active: Optional[bool] = None
    news_block: Optional[bool] = None
    signal_history: Optional[List[str]] = None
    session_preference: Optional[str] = None
    account_balance: Optional[float] = None
    account_equity: Optional[float] = None


class PredictResponse(BaseModel):
    asset: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    regime: str
    probability: float
    transition_probability: float
    cdi: float
    severity: int
    confirmed_elevated: bool
    consecutive_elevated: int
    risk_multiplier: float
    top_drift_feature: Optional[str]
    rolling_samples: int
    model_fallback: Optional[str] = None

    # Rich GoldDiggr fields
    schema_version: Optional[str] = None
    timestamp: Optional[str] = None
    connection_status: Optional[str] = None
    session: Optional[dict] = None
    market_state: Optional[str] = None
    market: Optional[dict] = None
    signal: Optional[dict] = None
    risk: Optional[dict] = None
    execution: Optional[dict] = None
    last_signal: Optional[str] = None
    signal_confidence: Optional[float] = None
    strategy_context: Optional[str] = None
    active_bias: Optional[str] = None
    spread_points: Optional[float] = None
    price: Optional[float] = None
    key_levels: Optional[dict] = None
    trade: Optional[dict] = None
    management: Optional[dict] = None
    diagnostics: Optional[dict] = None
    debug: Optional[dict] = None
    v3: Optional[dict] = None


def select_regime(asset: str, impulse_norm: float) -> str:
    metadata = asset_metadata.get(asset, {})
    return _select_regime(asset, impulse_norm, metadata)


def _ensure_loaded(asset: str) -> None:
    """
    Keep backward-compatible caches populated for the original tests.
    """
    if asset in asset_metadata:
        return

    try:
        bundle = load_asset_bundle(asset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"No models found for asset: {asset}") from exc

    asset_metadata[asset] = bundle.metadata
    asset_models[asset] = bundle.models
    asset_transition_models[asset] = bundle.transition_model
    asset_state[asset] = {"loaded": True}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "loaded_assets": loaded_assets(),
    }


@app.get("/metrics")
def metrics():
    return {
        "assets_loaded": len(loaded_assets()),
        "assets": loaded_assets(),
    }


@app.post("/analyze", response_model=PredictResponse)
def analyze(req: PredictRequest):
    try:
        if len(req.candles) == 0:
            raise HTTPException(status_code=400, detail="No candles provided")
        if len(req.candles) < 20:
            raise HTTPException(status_code=400, detail="Insufficient candles (need at least 20)")
        _ensure_loaded(req.asset)
        records = [c.dict(exclude_none=True) for c in req.candles]
        for r in records:
            if "timestamp" in r and "time" not in r:
                r["time"] = r.pop("timestamp")

        request_context = {
            "spread_points": req.spread_points,
            "price": req.ask if req.ask is not None else req.bid,
            "confidence_threshold": req.confidence_threshold or 0.65,
            "max_spread_points": req.max_spread_points or 80.0,
            "drawdown_pct": req.drawdown_pct or 0.0,
            "daily_loss_pct": req.daily_loss_pct or 0.0,
            "winning_streak": req.winning_streak or 0,
            "losing_streak": req.losing_streak or 0,
            "bars_since_last_trade": req.bars_since_last_trade or 999,
            "open_positions": req.open_positions or 0,
            "max_positions": req.max_positions or 1,
            "min_persistence": req.min_persistence or 2,
            "cooldown_active": bool(req.cooldown_active),
            "news_block": bool(req.news_block),
            "signal_history": req.signal_history or [],
            "session_preference": req.session_preference,
            "account_balance": req.account_balance,
            "account_equity": req.account_equity,
        }

        response = run_inference(
            asset=req.asset,
            timeframe=req.timeframe,
            candles=records,
            request_context=request_context,
        )
        return response
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Backward-compatible alias for the original endpoint.
    return analyze(req)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "detail": exc.detail},
    )
