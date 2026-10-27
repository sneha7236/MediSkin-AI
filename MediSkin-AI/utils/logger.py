"""Application logging. Never pass passwords, tokens or personal data to the logger."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import config

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("mediskin")
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    root.propagate = False
    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.LOGS_DIR / "mediskin.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("File logging unavailable; using console only.")

    for warning in config.CONFIG_WARNINGS:
        root.warning("Config: %s", warning)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(f"mediskin.{name}")
