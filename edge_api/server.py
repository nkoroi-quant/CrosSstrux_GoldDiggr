# edge_api/server.py - Production FastAPI server with pre-loading, API-key, optional rich fields
# Updated to modern lifespan (no more on_event deprecation)

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends, Query
import logging
from typing import Optional, Dict, Any

import sentry_sdk
from pydantic import BaseModel

from inference.engine import run_inference
from inference.loader import load_asset_bundle
from config.settings import settings

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


# Simple API key middleware
async def verify_api_key(request: Request):
    if settings.API_KEY and request.headers.get("X-API-Key") != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


class PredictRequest(BaseModel):
    asset: str
    candles: list[Dict[str, Any]]
    spread_points: Optional[float] = None
    include_rich: Optional[bool] = Query(False, description="Include full rich GoldDiggr payload")


@app.post("/analyze")
async def analyze(req: PredictRequest, _: bool = Depends(verify_api_key)):
    response = run_inference(
        asset=req.asset,
        timeframe="M1",
        candles=req.candles,
        request_context={"spread_points": req.spread_points},
    )
    if not req.include_rich:
        for k in ["session", "market", "signal", "trade", "levels"]:
            response.pop(k, None)
    return response


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
