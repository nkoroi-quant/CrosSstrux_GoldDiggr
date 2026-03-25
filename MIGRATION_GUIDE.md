# CrossStrux v1 -> v2 migration guide

## What changed

- CrosSstrux stays the intelligence core.
- GoldDiggr becomes the execution terminal and panel.
- The API now returns a rich decision object, not just BUY/SELL.
- XAUUSD is the production asset path.
- Model artifacts are regenerated locally, not shipped.

## Re-ingest data

1. Open MT5 and ensure the XAUUSD symbol is available.
2. Update `config/symbol_map.json` to match your broker symbol.
3. Run:

```bash
python -m data_layer.collector --assets XAUUSD
```

This creates:

```text
data/raw/XAUUSD_M1.parquet
```

## Generate training data

CrossStrux v2 builds features during training from the raw parquet file.

Train the model stack:

```bash
python run_training_pipeline.py --assets XAUUSD --force-retrain
```

This creates:

```text
models/XAUUSD/
├── metadata.json
├── low_model.pkl
├── mid_model.pkl
├── high_model.pkl
├── transition_model.pkl
├── drift_baseline_low.json
├── drift_baseline_mid.json
└── drift_baseline_high.json
```

## Start the server

```bash
uvicorn edge_api.server:app --host 0.0.0.0 --port 8000
```

## MT5 setup

1. Copy `mt5/GoldDiggr.mq5` into your MetaEditor Experts folder.
2. Compile it.
3. Allow WebRequest for your API host in MT5 options.
4. Attach the EA to your XAUUSD chart.

## Rich response fields used by GoldDiggr

- session in EAT
- connection status
- market state
- last signal + confidence
- strategy context
- active bias
- spread + price
- key levels
- trade blueprint
- management parameters

## Safe first run

Use a demo account first. Confirm:
- server health endpoint returns healthy
- `/predict` returns a valid JSON response
- the panel updates on the EA chart
- manual BUY/SELL/CLOSE ALL buttons work
