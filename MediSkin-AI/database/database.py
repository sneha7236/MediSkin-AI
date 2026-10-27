"""SQLAlchemy engine/session management (SQLite)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import config
from utils.logger import get_logger

logger = get_logger("database")

Base = declarative_base()

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def init_engine(database_url: str | None = None) -> Engine:
    """(Re)create the engine and ensure all tables exist. Safe to call repeatedly."""
    global _engine, _SessionFactory
    from database import models  # noqa: F401  (register tables on Base)

    url = database_url or config.DATABASE_URL
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(engine)
    if _engine is not None:
        _engine.dispose()
    _engine = engine
    _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return engine


def get_engine() -> Engine:
    return _engine or init_engine()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    if _SessionFactory is None:
        init_engine()
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database transaction failed; rolled back.")
        raise
    finally:
        session.close()
