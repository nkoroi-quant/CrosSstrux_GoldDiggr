# edge_api/server.py - FastAPI edge API for GoldDiggr / CrossStrux
# Supports H1+M15 context windows and M5+M1 entry windows.

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import logging

import sentry_sdk
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from inference.engine import run_inference
from inference.loader import load_asset_bundle

if settings.SENTRY_DSN:
    sentry_sdk.init(dsn=settings.SENTRY_DSN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading all models...")
    for asset in ["XAUUSD"]:
        load_asset_bundle(asset)
    logger.info("Models warm")
    yield
    logger.info("Shutting down...")


app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION, lifespan=lifespan)


async def verify_api_key(request: Request):
    if settings.API_KEY and request.headers.get("X-API-Key") != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


class PredictRequest(BaseModel):
    asset: str
    candles: List[Dict[str, Any]] = Field(default_factory=list)
    context_candles_h1: List[Dict[str, Any]] = Field(default_factory=list)
    context_candles_m15: List[Dict[str, Any]] = Field(default_factory=list)
    context_candles_m5: List[Dict[str, Any]] = Field(default_factory=list)
    entry_candles_m5: List[Dict[str, Any]] = Field(default_factory=list)
    entry_candles_m1: List[Dict[str, Any]] = Field(default_factory=list)
    signal_history: List[str] = Field(default_factory=list)
    spread_points: Optional[float] = None
    drawdown_pct: Optional[float] = None
    bars_since_last_trade: Optional[int] = None
    open_positions: Optional[int] = None
    max_positions: Optional[int] = None
    confidence_threshold: Optional[float] = None
    max_spread_points: Optional[float] = None
    news_block: bool = False
    cooldown_active: bool = False
    min_persistence: Optional[int] = None
    losing_streak: Optional[int] = None
    winning_streak: Optional[int] = None
    include_rich: bool = False


@app.post("/analyze")
async def analyze(req: PredictRequest, _: bool = Depends(verify_api_key)):
    try:
        context_candles_h1 = req.context_candles_h1 or []
        context_candles_m15 = req.context_candles_m15 or req.context_candles_m5 or req.candles
        entry_candles_m5 = req.entry_candles_m5 or req.context_candles_m5 or req.context_candles_m15 or context_candles_m15
        request_context = {
            "spread_points": req.spread_points,
            "drawdown_pct": req.drawdown_pct,
            "bars_since_last_trade": req.bars_since_last_trade,
            "open_positions": req.open_positions,
            "max_positions": req.max_positions,
            "confidence_threshold": req.confidence_threshold,
            "max_spread_points": req.max_spread_points,
            "signal_history": req.signal_history,
            "news_block": req.news_block,
            "cooldown_active": req.cooldown_active,
            "min_persistence": req.min_persistence,
            "losing_streak": req.losing_streak,
            "winning_streak": req.winning_streak,
            "context_candles_h1": context_candles_h1,
            "context_candles_m15": context_candles_m15,
            "context_candles_m5": req.context_candles_m5 or context_candles_m15,
            "entry_candles_m5": entry_candles_m5,
            "entry_candles_m1": req.entry_candles_m1,
        }
        response = run_inference(
            asset=req.asset,
            timeframe="H1_M15_CONTEXT__M5_M1_ENTRY",
            candles=context_candles_m15,
            request_context=request_context,
        )
        if not req.include_rich:
            for k in ["session", "market", "signal", "trade", "levels", "management", "diagnostics", "v3"]:
                response.pop(k, None)
        return response
    except Exception as exc:
        logger.exception("Analyze request failed")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})


@app.post("/predict")
async def predict(req: PredictRequest, _: bool = Depends(verify_api_key)):
    logger.warning("DEPRECATED: /predict endpoint - use /analyze instead")
    return await analyze(req)


@app.get("/warmup")
async def warmup():
    return {"status": "warm", "assets": ["XAUUSD"]}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": settings.APP_VERSION}
