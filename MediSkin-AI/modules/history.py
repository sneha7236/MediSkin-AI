"""Prediction history and dashboard statistics (UI-independent).

``Prediction.condition`` stores the class *slug* (e.g. ``mel``); use
``modules.disease_info.display_name`` to show it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import config, constants
from database import crud
from database.crud import PredictionRecord, ReportRecord
from modules.disease_info import display_name
from modules import report_generator
from prediction import confidence as confidence_rules
from utils import security
from utils.logger import get_logger

logger = get_logger("history")


@dataclass(frozen=True)
class DashboardStats:
    total: int
    latest: PredictionRecord | None
    recent: list[PredictionRecord]


def save_prediction(user_id: int, result: dict) -> PredictionRecord:
    """Persist a ``Predictor.predict`` result for ``user_id``."""
    return crud.create_prediction(
        user_id, result["condition_id"], result["confidence"], result["risk_level"], result["probabilities"]
    )


def list_history(user_id: int) -> list[PredictionRecord]:
    return crud.list_predictions(user_id)


def get_entry(user_id: int, prediction_id: int) -> PredictionRecord | None:
    return crud.get_prediction(user_id, prediction_id)


def dashboard_stats(user_id: int, recent_count: int = 5) -> DashboardStats:
    recent = crud.list_predictions(user_id, limit=recent_count)
    return DashboardStats(total=crud.count_predictions(user_id), latest=recent[0] if recent else None, recent=recent)


def _safe_report_path(file_path: str) -> Path | None:
    """The report path only if it lies inside the reports directory and exists."""
    path = Path(file_path)
    if security.is_within_directory(path, config.REPORTS_DIR) and path.is_file():
        return path
    return None


def delete_entry(user_id: int, prediction_id: int) -> bool:
    """Delete a prediction (user-scoped) and its report files. True if something was deleted."""
    if crud.get_prediction(user_id, prediction_id) is None:
        return False
    paths = crud.delete_prediction(user_id, prediction_id)
    crud.remove_files([str(p) for p in map(Path, paths) if security.is_within_directory(p, config.REPORTS_DIR)])
    logger.info("History entry deleted (user=%s)", user_id)
    return True


def get_report_for_prediction(user_id: int, prediction_id: int) -> ReportRecord | None:
    report = crud.get_latest_report(user_id, prediction_id)
    return report if report and _safe_report_path(report.file_path) else None


def read_report(report: ReportRecord) -> bytes | None:
    """PDF bytes for an already-authorised report record, or None if the file is unavailable."""
    path = _safe_report_path(report.file_path)
    if path is None:
        return None
    try:
        return path.read_bytes()
    except OSError:
        logger.error("Could not read report file")
        return None


def read_report_bytes(user_id: int, report_id: int) -> bytes | None:
    """PDF bytes for one of the user's reports, or None if missing/not theirs."""
    report = next((r for r in crud.list_reports(user_id) if r.id == report_id), None)
    return read_report(report) if report else None


def reports_by_prediction(user_id: int) -> dict[int, ReportRecord]:
    """Newest usable report per prediction id (one query, for list views)."""
    latest: dict[int, ReportRecord] = {}
    for report in crud.list_reports(user_id):  # newest first
        if report.prediction_id not in latest and _safe_report_path(report.file_path):
            latest[report.prediction_id] = report
    return latest


def ensure_report(user_id: int, user_name: str, prediction_id: int,
                  image: Image.Image | None = None) -> ReportRecord:
    """Return the prediction's existing report, generating and recording one if needed.

    Raises ``report_generator.ReportError`` / ``crud.DatabaseError`` with safe messages.
    """
    existing = get_report_for_prediction(user_id, prediction_id)
    if existing:
        return existing
    prediction = crud.get_prediction(user_id, prediction_id)
    if prediction is None:
        raise report_generator.ReportError("This analysis no longer exists.")
    return report_generator.create_and_store_report(user_id, user_name, prediction, image)


def result_from_record(record: PredictionRecord) -> dict:
    """Rebuild a result dict (same shape the UI uses) from a stored prediction."""
    probabilities = record.probabilities or {}
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return {
        "condition": display_name(record.condition),
        "condition_id": record.condition,
        "confidence": record.confidence,
        "confidence_band": confidence_rules.confidence_band(record.confidence),
        "risk_level": record.risk_level,
        "probabilities": probabilities,
        "top_predictions": [{"condition": name, "probability": prob}
                            for name, prob in ranked[: constants.TOP_K_PREDICTIONS]],
    }
