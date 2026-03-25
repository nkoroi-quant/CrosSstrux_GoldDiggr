# edge_api/server.py - Pre-load models, rate limiter stub, API-key middleware, optional rich fields, deprecated warning

from fastapi import FastAPI, HTTPException, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
from typing import Optional
import sentry_sdk
from pydantic import BaseModel

from inference.engine import run_inference
from inference.loader import load_asset_bundle, loaded_assets
from config.settings import settings

if settings.SENTRY_DSN:
    sentry_sdk.init(dsn=settings.SENTRY_DSN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)

# Simple API key middleware
async def verify_api_key(request: Request):
    if settings.API_KEY and request.headers.get("X-API-Key") != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Pre-load on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading all models...")
    for asset in ["XAUUSD"]:  # extend via settings later
        load_asset_bundle(asset)
    logger.info("Models warm")

class PredictRequest(BaseModel):
    # ... original + optional fields
    include_rich: Optional[bool] = Query(False, description="Include full GoldDiggr rich payload")

@app.post("/analyze")
async def analyze(req: PredictRequest, _: bool = Depends(verify_api_key)):
    # ... original logic
    response = run_inference(...)
    if not req.include_rich:
        # strip rich fields for backward
        for k in ["session", "market", "signal", "trade", ...]:
            response.pop(k, None)
    return response

@app.post("/predict")
async def predict(req: PredictRequest, _: bool = Depends(verify_api_key)):
    logger.warning("DEPRECATED: /predict - use /analyze instead")
    return await analyze(req)

@app.get("/warmup")
async def warmup():
    return {"status": "warm", "assets": loaded_assets()}