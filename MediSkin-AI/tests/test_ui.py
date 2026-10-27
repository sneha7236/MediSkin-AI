"""UI-layer helpers, history/report orchestration and an in-process application smoke test."""
import json

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from config import config
from database import crud
from modules import analytics, authentication as auth, history
from modules.hospital_recommendation import Hospital
from modules.report_generator import ReportError
from pages import hospitals as hospitals_page
from utils import ui

RESULT = {
    "condition": "Melanoma", "condition_id": "mel", "confidence": 0.82, "risk_level": "Higher concern",
    "probabilities": {"Melanoma": 0.82, "Melanocytic Nevus": 0.12, "Benign Keratosis": 0.04, "Vascular Lesion": 0.02},
}


@pytest.fixture()
def user(db):
    return crud.create_user("Test User", "ui@example.com", "hash")


@pytest.fixture()
def reports_dir(tmp_path, monkeypatch):
    directory = tmp_path / "reports"
    directory.mkdir()
    monkeypatch.setattr(config, "REPORTS_DIR", directory)
    return directory


# --- History / report orchestration -----------------------------------------
def test_result_from_record_rebuilds_ui_result(user):
    record = history.save_prediction(user.id, RESULT)
    rebuilt = history.result_from_record(record)
    assert rebuilt["condition_id"] == "mel" and rebuilt["confidence"] == pytest.approx(0.82)
    assert rebuilt["confidence_band"] == "High"
    assert [p["condition"] for p in rebuilt["top_predictions"]] == ["Melanoma", "Melanocytic Nevus", "Benign Keratosis"]


def test_ensure_report_is_idempotent_and_listed(user, reports_dir):
    record = history.save_prediction(user.id, RESULT)
    first = history.ensure_report(user.id, user.name, record.id, Image.new("RGB", (40, 40), (180, 120, 90)))
    second = history.ensure_report(user.id, user.name, record.id)
    assert first.id == second.id
    assert len(list(reports_dir.glob("*.pdf"))) == 1
    assert history.reports_by_prediction(user.id)[record.id].id == first.id
    assert history.read_report(first)[:5] == b"%PDF-"


def test_ensure_report_rejects_unknown_or_foreign_prediction(user, reports_dir):
    other = crud.create_user("Other", "other@example.com", "hash")
    foreign = history.save_prediction(other.id, RESULT)
    with pytest.raises(ReportError):
        history.ensure_report(user.id, user.name, foreign.id)
    with pytest.raises(ReportError):
        history.ensure_report(user.id, user.name, 9999)
    assert list(reports_dir.glob("*.pdf")) == []


def test_reports_by_prediction_skips_missing_files(user, reports_dir):
    record = history.save_prediction(user.id, RESULT)
    report = history.ensure_report(user.id, user.name, record.id)
    next(reports_dir.glob("*.pdf")).unlink()
    assert history.reports_by_prediction(user.id) == {}
    assert history.read_report(report) is None


# --- HTML helpers escape dynamic text ----------------------------------------
def test_html_helpers_escape_untrusted_text():
    assert "<script>" not in ui.risk_badge("<script>alert(1)</script>")
    assert "<img" not in ui.metric_card("<img src=x>", "<b>v</b>", "<i>h</i>")
    bars = ui.probability_bars([('<b>x</b>"', 0.5)])
    assert "<b>" not in bars and "50.0%" in bars


def test_probability_bars_clamp_out_of_range_values():
    bars = ui.probability_bars([("a", 1.7), ("b", -0.2)])
    assert "width:100.0%" in bars and "width:0.0%" in bars


def test_hospital_card_escapes_and_links_safely():
    card = hospitals_page._card(Hospital(
        name="<b>Evil</b>", city="Pune", address="1 <i>Road</i>", phone="<x>", website="https://example.org/?a=1&b=2",
        dermatology=True, distance_km=3.14159,
    ))
    assert "<b>Evil</b>" not in card and "&lt;b&gt;Evil" in card
    assert 'rel="noopener noreferrer"' in card and "&amp;b=2" in card
    assert "Dermatology available" in card and "3.1 km" in card


def test_hospital_card_unknown_dermatology_is_not_claimed():
    card = hospitals_page._card(Hospital(name="Clinic", city="Pune"))
    assert "availability unknown" in card


# --- Training curves ---------------------------------------------------------
def test_training_history_loaded_from_file(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({
        "stage1": {"accuracy": [0.4, 0.5], "val_accuracy": [0.3, 0.45], "loss": [1.5, 1.2], "val_loss": [1.6, 1.3]},
        "fine_tune": {"accuracy": [0.6], "val_accuracy": [0.5], "other": [9]},
    }))
    monkeypatch.setattr(config, "HISTORY_PATH", path)
    frame = analytics.load_training_history()
    assert list(frame.index) == [1, 2, 3]
    assert "Validation accuracy" in frame.columns and "other" not in frame.columns


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"stage1": "x"}', '{"stage1": {}}'])
def test_training_history_missing_or_malformed_is_none(tmp_path, monkeypatch, content):
    path = tmp_path / "history.json"
    if content is not None:
        path.write_text(content)
    monkeypatch.setattr(config, "HISTORY_PATH", path)
    assert analytics.load_training_history() is None


# --- Application smoke test (runs app.py in-process) ---------------------------
@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    st.cache_resource.clear()
    auth.reset_login_throttle()
    yield AppTest.from_file(str(config.BASE_DIR / "app.py"), default_timeout=90)
    st.cache_resource.clear()


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_app_starts_on_sign_in_page(app):
    app.run()
    assert not app.exception
    assert [b.label for b in app.button] == ["Sign in", "Create account"]
    assert auth.SESSION_USER_KEY not in app.session_state


def test_register_then_dashboard_then_sign_out(app):
    app.run()
    app.text_input(key="register_email").input("smoke@example.com")
    app.text_input(key="register_password").input("Str0ng!Passw0rd")
    next(t for t in app.text_input if t.label == "Full name").input("Smoke Tester")
    next(t for t in app.text_input if t.label == "Confirm password").input("Str0ng!Passw0rd")
    _button(app, "Create account").click().run()
    assert not app.exception
    assert app.session_state[auth.SESSION_USER_KEY]["name"] == "Smoke Tester"
    assert "password" not in str(app.session_state[auth.SESSION_USER_KEY]).lower()
    _button(app, "Sign out").click().run()
    assert not app.exception
    assert auth.SESSION_USER_KEY not in app.session_state


def test_weak_password_shows_error_and_creates_no_account(app):
    app.run()
    app.text_input(key="register_email").input("weak@example.com")
    app.text_input(key="register_password").input("weak")
    next(t for t in app.text_input if t.label == "Full name").input("Weak User")
    next(t for t in app.text_input if t.label == "Confirm password").input("weak")
    _button(app, "Create account").click().run()
    assert any("Password must contain" in e.value for e in app.error)
    assert crud.get_user_credentials_by_email("weak@example.com") is None


def test_wrong_login_shows_generic_error(app):
    app.run()
    next(t for t in app.text_input if t.label == "Email" and t.key != "register_email").input("nobody@example.com")
    next(t for t in app.text_input if t.label == "Password" and t.key != "register_password").input("Nope!Nope1")
    _button(app, "Sign in").click().run()
    assert [e.value for e in app.error] == [auth.GENERIC_LOGIN_ERROR]
