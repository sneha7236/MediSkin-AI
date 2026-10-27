"""Application configuration, loaded from environment variables (.env).

Invalid or missing values fall back to safe defaults; problems are collected in
``CONFIG_WARNINGS`` so the logger can report them at startup.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CONFIG_WARNINGS: list[str] = []


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        CONFIG_WARNINGS.append(f"{name} is not an integer; using default {default}.")
        return default
    if value < minimum:
        CONFIG_WARNINGS.append(f"{name} must be >= {minimum}; using default {default}.")
        return default
    return value


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


# --- Paths ---------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = _resolve(os.getenv("MODEL_PATH", "models/skin_model.keras"))
CLASS_NAMES_PATH = _resolve(os.getenv("CLASS_NAMES_PATH", "models/class_names.json"))
METRICS_PATH = _resolve(os.getenv("METRICS_PATH", "models/model_metrics.json"))
HISTORY_PATH = _resolve(os.getenv("HISTORY_PATH", "models/training_history.json"))
REPORTS_DIR = _resolve(os.getenv("REPORTS_DIR", "reports/generated"))
LOGS_DIR = BASE_DIR / "logs"
STYLE_PATH = BASE_DIR / "assets" / "styles" / "style.css"
HOSPITALS_DATA_PATH = METADATA_DIR / "hospitals.json"

# --- Database ------------------------------------------------------------
_db_url = os.getenv("DATABASE_URL", "sqlite:///medi_skin.db")
if _db_url.startswith("sqlite:///") and not _db_url.startswith("sqlite:////") and ":memory:" not in _db_url:
    # Make relative SQLite paths independent of the working directory.
    _db_url = "sqlite:///" + str(BASE_DIR / _db_url[len("sqlite:///"):])
DATABASE_URL = _db_url

# --- Upload / session ----------------------------------------------------
MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 5)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
SESSION_TIMEOUT_MINUTES = _int_env("SESSION_TIMEOUT_MINUTES", 30)

# --- Logging -------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Hospital provider (future API hook) ---------------------------------
HOSPITAL_PROVIDER = os.getenv("HOSPITAL_PROVIDER", "local").strip().lower()
HOSPITAL_API_KEY = os.getenv("HOSPITAL_API_KEY", "")  # never logged or displayed


def ensure_runtime_dirs() -> None:
    """Create directories the app writes to."""
    for directory in (REPORTS_DIR, LOGS_DIR, MODELS_DIR, METADATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
