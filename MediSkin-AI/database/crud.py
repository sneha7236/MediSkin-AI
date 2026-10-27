"""Database access functions. All reads/writes of user data are scoped by user_id.

Functions return plain dataclasses (``*Record``) so the UI never touches ORM objects
and password hashes never leave the authentication layer unintentionally.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from database.database import session_scope
from database.models import Feedback, Prediction, Report, User
from utils.logger import get_logger

logger = get_logger("crud")


class DatabaseError(RuntimeError):
    """Wraps low-level DB failures with a safe message."""


class DuplicateEmailError(DatabaseError):
    pass


@dataclass(frozen=True)
class UserRecord:
    id: int
    name: str
    email: str
    created_at: datetime


@dataclass(frozen=True)
class UserCredentials:
    id: int
    name: str
    email: str
    password_hash: str
    created_at: datetime


@dataclass(frozen=True)
class PredictionRecord:
    id: int
    user_id: int
    condition: str
    confidence: float
    risk_level: str
    probabilities: dict[str, float]
    created_at: datetime


@dataclass(frozen=True)
class ReportRecord:
    id: int
    user_id: int
    prediction_id: int
    file_path: str
    created_at: datetime


@dataclass(frozen=True)
class FeedbackRecord:
    id: int
    user_id: int
    prediction_id: int | None
    rating: int
    helpful: bool | None
    comment: str
    created_at: datetime


def _guard(func_):
    """Decorator: convert SQLAlchemy errors into DatabaseError (logged, no details to UI)."""
    def wrapper(*args, **kwargs):
        try:
            return func_(*args, **kwargs)
        except DatabaseError:
            raise
        except SQLAlchemyError as exc:
            logger.error("Database error in %s: %s", func_.__name__, type(exc).__name__)
            raise DatabaseError("A database error occurred. Please try again.") from exc
    wrapper.__name__ = func_.__name__
    wrapper.__doc__ = func_.__doc__
    return wrapper


def _user_record(row: User) -> UserRecord:
    return UserRecord(row.id, row.name, row.email, row.created_at)


def _prediction_record(row: Prediction) -> PredictionRecord:
    try:
        probabilities = json.loads(row.probabilities_json or "{}")
    except json.JSONDecodeError:
        probabilities = {}
    return PredictionRecord(
        row.id, row.user_id, row.condition, row.confidence, row.risk_level,
        probabilities, row.created_at,
    )


def _report_record(row: Report) -> ReportRecord:
    return ReportRecord(row.id, row.user_id, row.prediction_id, row.file_path, row.created_at)


# --- Users ---------------------------------------------------------------
@_guard
def create_user(name: str, email: str, password_hash: str) -> UserRecord:
    try:
        with session_scope() as session:
            user = User(name=name, email=email, password_hash=password_hash)
            session.add(user)
            session.flush()
            return _user_record(user)
    except IntegrityError as exc:
        raise DuplicateEmailError("An account with this email already exists.") from exc


@_guard
def get_user_credentials_by_email(email: str) -> UserCredentials | None:
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None:
            return None
        return UserCredentials(user.id, user.name, user.email, user.password_hash, user.created_at)


@_guard
def get_user(user_id: int) -> UserRecord | None:
    with session_scope() as session:
        user = session.get(User, user_id)
        return _user_record(user) if user else None


@_guard
def get_user_credentials_by_id(user_id: int) -> UserCredentials | None:
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            return None
        return UserCredentials(user.id, user.name, user.email, user.password_hash, user.created_at)


@_guard
def update_password_hash(user_id: int, password_hash: str) -> bool:
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            return False
        user.password_hash = password_hash
        return True


@_guard
def update_user_name(user_id: int, name: str) -> UserRecord | None:
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            return None
        user.name = name
        return _user_record(user)


# --- Predictions ---------------------------------------------------------
@_guard
def create_prediction(
    user_id: int, condition: str, confidence: float, risk_level: str,
    probabilities: dict[str, float],
) -> PredictionRecord:
    with session_scope() as session:
        row = Prediction(
            user_id=user_id, condition=condition, confidence=float(confidence),
            risk_level=risk_level, probabilities_json=json.dumps(probabilities),
        )
        session.add(row)
        session.flush()
        return _prediction_record(row)


@_guard
def get_prediction(user_id: int, prediction_id: int) -> PredictionRecord | None:
    with session_scope() as session:
        row = session.scalar(
            select(Prediction).where(Prediction.id == prediction_id, Prediction.user_id == user_id)
        )
        return _prediction_record(row) if row else None


@_guard
def list_predictions(user_id: int, limit: int | None = None) -> list[PredictionRecord]:
    with session_scope() as session:
        stmt = (
            select(Prediction).where(Prediction.user_id == user_id)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        return [_prediction_record(r) for r in session.scalars(stmt)]


@_guard
def count_predictions(user_id: int) -> int:
    with session_scope() as session:
        return session.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.user_id == user_id)
        ) or 0


@_guard
def delete_prediction(user_id: int, prediction_id: int) -> list[str]:
    """Delete a prediction (and cascaded reports/feedback).

    Returns the file paths of its reports so the caller can remove the files.
    """
    with session_scope() as session:
        row = session.scalar(
            select(Prediction).where(Prediction.id == prediction_id, Prediction.user_id == user_id)
        )
        if row is None:
            return []
        paths = [r.file_path for r in row.reports]
        session.delete(row)
        return paths


# --- Reports -------------------------------------------------------------
@_guard
def create_report(user_id: int, prediction_id: int, file_path: str) -> ReportRecord:
    with session_scope() as session:
        owns = session.scalar(
            select(Prediction.id).where(Prediction.id == prediction_id, Prediction.user_id == user_id)
        )
        if owns is None:
            raise DatabaseError("Prediction not found.")
        row = Report(user_id=user_id, prediction_id=prediction_id, file_path=file_path)
        session.add(row)
        session.flush()
        return _report_record(row)


@_guard
def get_latest_report(user_id: int, prediction_id: int) -> ReportRecord | None:
    with session_scope() as session:
        row = session.scalar(
            select(Report).where(Report.user_id == user_id, Report.prediction_id == prediction_id)
            .order_by(Report.created_at.desc(), Report.id.desc())
        )
        return _report_record(row) if row else None


@_guard
def list_reports(user_id: int) -> list[ReportRecord]:
    with session_scope() as session:
        rows = session.scalars(
            select(Report).where(Report.user_id == user_id)
            .order_by(Report.created_at.desc(), Report.id.desc())
        )
        return [_report_record(r) for r in rows]


# --- Feedback ------------------------------------------------------------
@_guard
def create_feedback(
    user_id: int, rating: int, comment: str, helpful: bool | None, prediction_id: int | None,
) -> FeedbackRecord:
    with session_scope() as session:
        if prediction_id is not None:
            owns = session.scalar(
                select(Prediction.id).where(Prediction.id == prediction_id, Prediction.user_id == user_id)
            )
            if owns is None:
                raise DatabaseError("Prediction not found.")
        row = Feedback(
            user_id=user_id, prediction_id=prediction_id, rating=rating,
            helpful=helpful, comment=comment,
        )
        session.add(row)
        session.flush()
        return FeedbackRecord(
            row.id, row.user_id, row.prediction_id, row.rating, row.helpful, row.comment, row.created_at
        )


@_guard
def list_feedback(user_id: int) -> list[FeedbackRecord]:
    with session_scope() as session:
        rows = session.scalars(
            select(Feedback).where(Feedback.user_id == user_id).order_by(Feedback.created_at.desc())
        )
        return [
            FeedbackRecord(r.id, r.user_id, r.prediction_id, r.rating, r.helpful, r.comment, r.created_at)
            for r in rows
        ]


def remove_files(paths: list[str]) -> None:
    """Best-effort removal of report files after DB rows are deleted."""
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete report file: %s", type(exc).__name__)
