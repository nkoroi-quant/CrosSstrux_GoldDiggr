# CrossStrux v2

CrossStrux v2 keeps the original XAUUSD intelligence stack and adds the
GoldDiggr execution adapter.

## What is inside

- deterministic structure detection
- XAUUSD regime thresholds and ML continuation models
- transition model
- drift governance
- rich API responses for GoldDiggr
- MT5 execution terminal with panel, toggles, manual buttons, and trade management

## Recommended flow

1. Ingest MT5 XAUUSD candles into `data/raw/XAUUSD_M1.parquet`
2. Train the XAUUSD model stack
3. Start the FastAPI server
4. Attach `mt5/GoldDiggr.mq5` in MetaTrader 5

## Quick start

```bash
pip install -r requirements.txt
python -m data_layer.collector --assets XAUUSD
python run_training_pipeline.py --assets XAUUSD --force-retrain
uvicorn edge_api.server:app --host 0.0.0.0 --port 8000
```

## MT5 mapping

Edit `config/symbol_map.json` so your broker symbol matches the logical asset.

Example:

```json
{"XAUUSD": "GOLD"}
```

If your broker uses `XAUUSD` directly, change the value to `XAUUSD`.

## API endpoints

- `GET /health`
- `GET /metrics`
- `POST /analyze`
- `POST /predict` (backward-compatible alias)

## Notes

This package does not ship trained models or raw data.
Use the training guide to rebuild them from your MT5 history.
