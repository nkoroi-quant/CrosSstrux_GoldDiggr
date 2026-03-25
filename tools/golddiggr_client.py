# tools/golddiggr_client.py - Lightweight Python client

import httpx
from typing import List, Dict


class GoldDiggrClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None):
        self.client = httpx.AsyncClient(timeout=10.0)
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key} if api_key else {}

    async def analyze(self, asset: str, candles: List[Dict], **kwargs):
        payload = {"asset": asset, "candles": candles, **kwargs}
        r = await self.client.post(f"{self.base_url}/analyze", json=payload, headers=self.headers)
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self.client.aclose()
