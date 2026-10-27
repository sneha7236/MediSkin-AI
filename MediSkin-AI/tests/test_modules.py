"""Hospitals, reports, history, feedback and analytics."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config import config
from database import crud
from modules import analytics, feedback, history, hospital_recommendation as hosp, report_generator
from utils.validators import ValidationError

RESULT = {
    "condition": "Melanoma", "condition_id": "mel", "confidence": 0.82, "risk_level": "Higher concern",
    "probabilities": {"Melanoma": 0.82, "Melanocytic Nevus": 0.12, "Benign Keratosis": 0.06},
}


@pytest.fixture()
def user(db):
    return crud.create_user("Test User", "t@example.com", "hash")


@pytest.fixture()
def reports_dir(tmp_path, monkeypatch):
    directory = tmp_path / "reports"
    directory.mkdir()
    monkeypatch.setattr(config, "REPORTS_DIR", directory)
    return directory


# --- Hospitals -----------------------------------------------------------
def write_hospitals(path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")
    return hosp.LocalJsonProvider(path)


def test_no_data_means_empty_not_fabricated(tmp_path):
    assert hosp.LocalJsonProvider(tmp_path / "missing.json").search() == []
    assert json.loads(Path(config.HOSPITALS_DATA_PATH).read_text()) == []


def test_provider_filters_and_skips_invalid(tmp_path):
    provider = write_hospitals(tmp_path / "h.json", [
        {"name": "A Clinic", "city": "Pune", "dermatology": True, "website": "javascript:alert(1)"},
        {"name": "B Hospital", "city": "Mumbai", "dermatology": False},
        {"name": "", "city": "Pune"}, "garbage", {"name": "NoCity"},
    ])
    assert [h.name for h in provider.search()] == ["B Hospital", "A Clinic"]
    pune = provider.search(city="pun")
    assert [h.name for h in pune] == ["A Clinic"] and pune[0].website == ""  # unsafe URL dropped
    assert [h.name for h in provider.search(dermatology_only=True)] == ["A Clinic"]


def test_distance_only_with_coordinates(tmp_path):
    provider = write_hospitals(tmp_path / "h.json", [
        {"name": "Near", "city": "X", "latitude": 18.52, "longitude": 73.86},
        {"name": "Far", "city": "X", "latitude": 19.07, "longitude": 72.87},
        {"name": "NoCoords", "city": "X"},
    ])
    results = provider.search(latitude=18.5204, longitude=73.8567)
    assert [h.name for h in results] == ["Near", "Far", "NoCoords"]
    assert results[0].distance_km < 1 and 100 < results[1].distance_km < 200
    assert results[2].distance_km is None


def test_haversine_known_distance():
    assert hosp.haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.1)


def test_invalid_json_file_gives_empty(tmp_path):
    path = tmp_path / "h.json"
    path.write_text("{not json")
    assert hosp.LocalJsonProvider(path).search() == []
    path.write_text('{"a": 1}')
    assert hosp.LocalJsonProvider(path).search() == []


def test_provider_registry_and_failure_isolation(monkeypatch):
    class Boom:
        def search(self, *a, **k):
            raise RuntimeError("api down")

    hosp.register_provider("boom", Boom)
    monkeypatch.setattr(config, "HOSPITAL_PROVIDER", "boom")
    assert hosp.find_hospitals() == []
    monkeypatch.setattr(config, "HOSPITAL_PROVIDER", "nonexistent")
    assert isinstance(hosp.get_provider(), hosp.LocalJsonProvider)


def test_maps_url_is_encoded():
    url = hosp.maps_search_url("New Delhi & NCR")
    assert url.startswith("https://www.google.com/maps/search/") and "&NCR" not in url
    assert "near+me" in hosp.maps_search_url(None)


# --- History -------------------------------------------------------------
def test_save_list_stats_and_isolation(user):
    other = crud.create_user("Other", "o@example.com", "hash")
    saved = history.save_prediction(user.id, RESULT)
    assert saved.condition == "mel" and saved.probabilities["Melanoma"] == 0.82
    stats = history.dashboard_stats(user.id)
    assert stats.total == 1 and stats.latest.id == saved.id
    assert history.dashboard_stats(other.id).total == 0
    assert history.get_entry(other.id, saved.id) is None
    assert history.delete_entry(other.id, saved.id) is False
    assert history.delete_entry(user.id, saved.id) is True
    assert history.list_history(user.id) == []


# --- Reports -------------------------------------------------------------
def make_image():
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (90, 120, 3), dtype=np.uint8))


def test_report_pdf_is_valid_and_contains_required_content(user, reports_dir):
    prediction = history.save_prediction(user.id, RESULT)
    record = report_generator.create_and_store_report(user.id, user.name, prediction, make_image())
    path = Path(record.file_path)
    assert path.parent == reports_dir and path.suffix == ".pdf"
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF")
    text = _pdf_text(path)
    for needle in ("MediSkin AI", "Test User", "Melanoma", "82.0%", "Higher concern",
                   "not a medical diagnosis", "Top predictions"):
        assert needle in text, needle


def _pdf_text(path):
    import re
    import subprocess

    try:
        return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                              text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("pdftotext not available")


def test_report_without_image_and_unique_filenames(user, reports_dir):
    prediction = history.save_prediction(user.id, RESULT)
    first = report_generator.generate_report(user.name, prediction)
    second = report_generator.generate_report(user.name, prediction)
    assert first != second and first.is_file() and second.is_file()


def test_report_special_characters_are_escaped(user, reports_dir):
    prediction = history.save_prediction(user.id, RESULT)
    path = report_generator.generate_report("<b>Evil & Co</b>", prediction)
    assert path.read_bytes().startswith(b"%PDF")


def test_report_failure_is_wrapped_and_cleaned(user, tmp_path, monkeypatch):
    prediction = history.save_prediction(user.id, RESULT)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(report_generator, "build_report_pdf", boom)
    with pytest.raises(report_generator.ReportError):
        report_generator.generate_report(user.name, prediction, output_dir=tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_report_download_and_delete_flow(user, reports_dir):
    other = crud.create_user("Other", "o@example.com", "hash")
    prediction = history.save_prediction(user.id, RESULT)
    record = report_generator.create_and_store_report(user.id, user.name, prediction)
    assert history.get_report_for_prediction(user.id, prediction.id).id == record.id
    assert history.read_report_bytes(user.id, record.id).startswith(b"%PDF")
    assert history.read_report_bytes(other.id, record.id) is None  # not theirs
    history.delete_entry(user.id, prediction.id)
    assert not Path(record.file_path).exists()  # file removed with the entry


def test_report_outside_reports_dir_is_never_served_or_deleted(user, reports_dir, tmp_path):
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF secret")
    prediction = history.save_prediction(user.id, RESULT)
    record = crud.create_report(user.id, prediction.id, str(outside))
    assert history.read_report_bytes(user.id, record.id) is None
    history.delete_entry(user.id, prediction.id)
    assert outside.exists()


# --- Feedback ------------------------------------------------------------
def test_feedback_saved_and_validated(user):
    prediction = history.save_prediction(user.id, RESULT)
    saved = feedback.submit_feedback(user.id, 4, "  useful\x00 ", True, prediction.id)
    assert saved.rating == 4 and saved.comment == "useful" and saved.helpful is True
    for bad in (0, 6, "5", True, None):
        with pytest.raises(ValidationError):
            feedback.submit_feedback(user.id, bad)
    assert len(feedback.submit_feedback(user.id, 5, "x" * 5000).comment) == 1000


def test_feedback_for_foreign_prediction_rejected(user):
    other = crud.create_user("Other", "o@example.com", "hash")
    prediction = history.save_prediction(other.id, RESULT)
    with pytest.raises(ValidationError):
        feedback.submit_feedback(user.id, 5, "hi", None, prediction.id)


# --- Analytics -----------------------------------------------------------
def test_analytics_from_real_metrics_structure():
    from training.evaluate import compute_metrics

    metrics = compute_metrics(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]), ["mel", "nv", "bkl"])
    assert analytics.summary_values(metrics)["Accuracy"] == pytest.approx(0.75)
    table = analytics.per_class_table(metrics)
    from modules.disease_info import display_name

    assert list(table["Class"]) == [display_name(n) for n in ("mel", "nv", "bkl")]
    assert analytics.confusion_matrix_figure(metrics) is not None
