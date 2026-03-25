from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    APP_TITLE: str = "CrossStrux GoldDiggr"
    APP_VERSION: str = "3.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    DEBUG: bool = False

    MODEL_ROOT: str = "models"
    DATA_DIR: str = "data/raw"
    CACHE_TTL_SECONDS: int = 3600
    MAX_CANDLES_CACHE: int = 500
    INCREMENTAL_WINDOW: int = 120

    API_KEY: Optional[str] = None
    SENTRY_DSN: Optional[str] = None

    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.65
    MAX_SPREAD_POINTS: float = 80.0
    PSI_ALERT_THRESHOLD: float = 0.15
    MIN_DRIFT_SAMPLES: int = 50

    SYMBOL_MAP: Dict[str, str] = {"XAUUSD": "GOLD"}


settings = Settings()
