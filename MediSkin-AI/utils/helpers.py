"""Small shared helpers."""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite stores naive datetimes)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def format_datetime(value: datetime | None) -> str:
    return value.strftime("%d %b %Y, %H:%M") if value else "—"


def percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"
