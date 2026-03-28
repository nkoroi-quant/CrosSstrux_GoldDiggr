# CrosSstrux GoldDiggr v3.2

Production-grade XAUUSD regime detection + structure trading engine with FastAPI backend and MT5 GoldDiggr EA.

## Architecture (Mermaid)

```mermaid
flowchart TD
    A[MT5 Terminal] --> B[GoldDiggr.mq5 EA v11.5]
    B --> C[POST /analyze<br>Rate-limited + Cached]
    D[Collector] --> E[Parquet M1]
    F[Training Pipeline] --> G[Model Registry + Drift Baseline]
    E --> F
    G --> H[Inference Engine Incremental + Numpy]
    H --> C
    C --> B
    I[FastAPI Server v3.2] --> J[Rate-Limited + API-Key + Sentry + LRU Cache]
    subgraph Dashboard
        K[Drift Monitor Streamlit]
    end