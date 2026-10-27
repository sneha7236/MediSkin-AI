"""Hospital / dermatology-clinic recommendations behind a replaceable provider interface.

The application ships with NO hospital records (none are invented). The default ``local``
provider reads real entries you add to ``data/metadata/hospitals.json``:

    [
      {"name": "...", "city": "...", "address": "...", "phone": "...", "website": "https://...",
       "dermatology": true, "latitude": 12.34, "longitude": 56.78}
    ]

Only ``name`` and ``city`` are required; every other field is optional. Distance is computed
only when both the search origin and the entry have coordinates.

To connect a live API later, implement ``HospitalProvider.search`` and call
``register_provider("my_api", MyProvider)``; then set ``HOSPITAL_PROVIDER=my_api`` in ``.env``.
Nothing else in the application changes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote_plus

from config import config
from utils.logger import get_logger

logger = get_logger("hospitals")


@dataclass(frozen=True)
class Hospital:
    name: str
    city: str
    address: str = ""
    phone: str = ""
    website: str = ""
    dermatology: bool | None = None  # None = unknown
    latitude: float | None = None
    longitude: float | None = None
    distance_km: float | None = None


class HospitalProvider(Protocol):
    def search(
        self, city: str | None = None, latitude: float | None = None,
        longitude: float | None = None, dermatology_only: bool = False, limit: int = 20,
    ) -> list[Hospital]: ...


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _coordinate(value, limit: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if -limit <= value <= limit else None


def _safe_website(value) -> str:
    url = _text(value)
    return url if url.lower().startswith(("http://", "https://")) else ""


def parse_hospital(entry) -> Hospital | None:
    """Convert one JSON object to a Hospital; None if it lacks a name/city."""
    if not isinstance(entry, dict):
        return None
    name, city = _text(entry.get("name")), _text(entry.get("city"))
    if not name or not city:
        return None
    derm = entry.get("dermatology")
    return Hospital(
        name=name, city=city, address=_text(entry.get("address")), phone=_text(entry.get("phone")),
        website=_safe_website(entry.get("website")),
        dermatology=derm if isinstance(derm, bool) else None,
        latitude=_coordinate(entry.get("latitude"), 90), longitude=_coordinate(entry.get("longitude"), 180),
    )


class LocalJsonProvider:
    """Reads curated hospitals from a JSON file."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.HOSPITALS_DATA_PATH)

    def _load(self) -> list[Hospital]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.error("Hospital data file is unreadable or invalid JSON")
            return []
        if not isinstance(raw, list):
            logger.error("Hospital data file must contain a JSON list")
            return []
        hospitals = [h for h in map(parse_hospital, raw) if h]
        if len(hospitals) != len(raw):
            logger.warning("Skipped %d invalid hospital entries", len(raw) - len(hospitals))
        return hospitals

    def search(self, city=None, latitude=None, longitude=None, dermatology_only=False, limit=20):
        results = self._load()
        if city and city.strip():
            needle = city.strip().lower()
            results = [h for h in results if needle in h.city.lower()]
        if dermatology_only:
            results = [h for h in results if h.dermatology is True]
        if latitude is not None and longitude is not None:
            results = [
                Hospital(**{**h.__dict__, "distance_km": haversine_km(latitude, longitude, h.latitude, h.longitude)})
                if h.latitude is not None and h.longitude is not None else h
                for h in results
            ]
            results.sort(key=lambda h: (h.distance_km is None, h.distance_km or 0.0, h.name.lower()))
        else:
            results.sort(key=lambda h: (h.city.lower(), h.name.lower()))
        return results[: max(1, limit)]


_PROVIDERS: dict[str, type] = {"local": LocalJsonProvider}


def register_provider(name: str, provider_cls: type) -> None:
    """Hook for a future live-API provider."""
    _PROVIDERS[name.strip().lower()] = provider_cls


def get_provider(name: str | None = None) -> HospitalProvider:
    name = (name or config.HOSPITAL_PROVIDER or "local").lower()
    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        logger.warning("Unknown hospital provider '%s'; using local data", name)
        provider_cls = LocalJsonProvider
    return provider_cls()


def find_hospitals(city: str | None = None, latitude: float | None = None,
                   longitude: float | None = None, dermatology_only: bool = False,
                   limit: int = 20) -> list[Hospital]:
    """Search via the configured provider. Provider failures yield an empty list, never a crash."""
    try:
        return get_provider().search(city, latitude, longitude, dermatology_only, limit)
    except Exception:
        logger.exception("Hospital provider failed")
        return []


def maps_search_url(city: str | None = None, query: str = "dermatologist") -> str:
    """A plain web-map search link (no API key, no data stored) for finding nearby care."""
    text = f"{query} in {city.strip()}" if city and city.strip() else f"{query} near me"
    return "https://www.google.com/maps/search/?api=1&query=" + quote_plus(text)
