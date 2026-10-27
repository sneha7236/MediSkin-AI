"""Shared Streamlit UI helpers: styling, HTML snippets, navigation and the cached model loader.

Only the ``pages`` package and ``app.py`` import this module; business logic stays in
``modules/``, ``prediction/`` and ``database/``.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import config, constants
from modules import authentication as auth
from prediction.predictor import Predictor, get_model_status
from utils.helpers import format_datetime
from utils.logger import get_logger

logger = get_logger("ui")

PAGES_KEY = "_nav_pages"
CURRENT_RESULT_KEY = "current_result"


def esc(value) -> str:
    """HTML-escape any dynamic text before it is placed in a custom HTML block."""
    return html.escape(str(value), quote=True)


def render_html(markup: str) -> None:
    st.html(markup)


# --- Styling -----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_css(path: str, modified: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_css() -> None:
    path = config.STYLE_PATH
    try:
        css = _read_css(str(path), path.stat().st_mtime)
    except OSError:
        logger.warning("Stylesheet not found; using default Streamlit styling.")
        return
    # st.html strips <style> tags; st.markdown with unsafe_allow_html keeps them.
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hide_sidebar() -> None:
    st.markdown(
        '<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],'
        '[data-testid="stExpandSidebarButton"] { display: none !important; }</style>',
        unsafe_allow_html=True,
    )


# --- HTML building blocks --------------------------------------------------------
def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    icon_html = f'<div class="icon"><span class="material-symbols-rounded">{esc(icon)}</span></div>' if icon else ""
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    render_html(f'<div class="ms-page-head">{icon_html}<div><h1>{esc(title)}</h1>{sub}</div></div>')


def disclaimer() -> None:
    render_html(
        f'<div class="ms-disclaimer" role="note"><strong>Medical disclaimer.</strong> '
        f"{esc(constants.MEDICAL_DISCLAIMER)}</div>"
    )


def risk_badge(risk_level: str) -> str:
    color = constants.RISK_COLORS.get(risk_level, "#475569")
    return f'<span class="ms-badge" style="--c:{esc(color)}">{esc(risk_level)}</span>'


def band_badge(band: str) -> str:
    color = {
        constants.CONFIDENCE_HIGH: "#0B6169",
        constants.CONFIDENCE_MODERATE: "#B45309",
        constants.CONFIDENCE_LOW: "#475569",
    }.get(band, "#475569")
    return f'<span class="ms-badge" style="--c:{color}">{esc(band)} confidence</span>'


def metric_card(label: str, value: str, hint: str = "", small: bool = False) -> str:
    size = " small" if small else ""
    hint_html = f'<div class="hint">{esc(hint)}</div>' if hint else ""
    return (f'<div class="ms-metric"><div class="label">{esc(label)}</div>'
            f'<div class="value{size}">{esc(value)}</div>{hint_html}</div>')


def probability_bars(items: list[tuple[str, float]]) -> str:
    rows = []
    for position, (name, probability) in enumerate(items):
        width = max(0.0, min(100.0, probability * 100))
        first = " first" if position == 0 else ""
        rows.append(
            f'<div class="ms-bar-row{first}"><div class="top"><span class="name">{esc(name)}</span>'
            f'<span class="val">{width:.1f}%</span></div>'
            f'<div class="ms-bar" role="img" aria-label="{esc(name)} {width:.1f} percent">'
            f'<span style="width:{width:.1f}%"></span></div></div>'
        )
    return "".join(rows)


def empty_state(title: str, message: str, icon: str = "inbox") -> None:
    render_html(
        f'<div class="ms-empty"><span class="material-symbols-rounded">{esc(icon)}</span>'
        f"<h3>{esc(title)}</h3><p>{esc(message)}</p></div>"
    )


def fmt_time(value: datetime | None) -> str:
    return f"{format_datetime(value)} UTC" if value else "—"


# --- Navigation / session -----------------------------------------------------------
def register_pages(pages: dict) -> None:
    st.session_state[PAGES_KEY] = pages


def go(key: str) -> None:
    """Switch to a registered page by key (see ``app.py``)."""
    st.switch_page(st.session_state[PAGES_KEY][key])


def current_user() -> dict:
    """The logged-in user (app.py guarantees a valid session before any page renders)."""
    return st.session_state[auth.SESSION_USER_KEY]


def set_current_result(prediction_id: int | None, result: dict, image=None) -> None:
    st.session_state[CURRENT_RESULT_KEY] = {
        "prediction_id": prediction_id, "result": result, "image": image, "gradcam": "pending",
    }


def get_current_result() -> dict | None:
    return st.session_state.get(CURRENT_RESULT_KEY)


def clear_current_result() -> None:
    st.session_state.pop(CURRENT_RESULT_KEY, None)


# --- Model loading (cached once per process) ------------------------------------------
@st.cache_resource(show_spinner="Loading AI model…")
def _load_predictor(model_modified: float, names_modified: float):
    """Keyed on file modification times, so replacing the model files reloads automatically."""
    from prediction.predictor import PredictionError

    try:
        return Predictor.load(), None
    except PredictionError as exc:
        return None, str(exc)
    except Exception:
        logger.exception("Unexpected error while loading the model")
        return None, "The AI model could not be loaded."


def get_predictor() -> tuple[Predictor | None, str | None]:
    """(predictor, None) when the model is usable, otherwise (None, user-friendly message)."""
    available, message = get_model_status()
    if not available:
        return None, message
    try:
        return _load_predictor(config.MODEL_PATH.stat().st_mtime, config.CLASS_NAMES_PATH.stat().st_mtime)
    except OSError:
        return None, message
