"""
Trading session helpers.

GoldDiggr displays the current session in EAT.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta


EAT_OFFSET = timedelta(hours=3)


def current_eat_time() -> datetime:
    return datetime.now(timezone.utc) + EAT_OFFSET


def session_label_for_hour(hour: int) -> str:
    if 0 <= hour < 7:
        return "Asia"
    if 7 <= hour < 13:
        return "London"
    if 13 <= hour < 18:
        return "New York"
    return "Off-hours"


def get_session_context() -> dict:
    dt = current_eat_time()
    return {
        "eat_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "session": session_label_for_hour(dt.hour),
        "timezone": "EAT",
    }
