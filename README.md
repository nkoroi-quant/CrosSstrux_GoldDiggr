# CrosSstrux GoldDiggr v3.1

Production-grade XAUUSD regime detection + structure trading engine with FastAPI backend and MT5 GoldDiggr EA.

## Architecture (Mermaid)

```mermaid
flowchart TD
    A[MT5 Terminal] --> B[GoldDiggr.mq5 EA]
    B --> C[POST /analyze]
    D[Collector] --> E[Parquet M1]
    F[Training Pipeline] --> G[Model Registry + Drift Baseline]
    E --> F
    G --> H[Inference Engine\nIncremental + Numpy]
    H --> C
    C --> B
    I[FastAPI Server] --> J[Rate-Limited + API-Key + Sentry]
    subgraph Dashboard
        K[Drift Monitor Streamlit]
    end