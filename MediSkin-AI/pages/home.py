"""Home dashboard: welcome, primary call to action, statistics, recent analyses, education."""
from __future__ import annotations

import streamlit as st

from config import constants
from database import crud
from database.crud import DatabaseError
from modules import history as history_service
from modules.disease_info import display_name
from prediction.predictor import get_model_status
from utils import ui
from utils.helpers import percent


def _recent_table(records) -> str:
    rows = "".join(
        f"<tr><td>{ui.esc(ui.fmt_time(r.created_at))}</td>"
        f'<td class="ms-row-label">{ui.esc(display_name(r.condition))}</td>'
        f"<td>{ui.esc(percent(r.confidence))}</td><td>{ui.risk_badge(r.risk_level)}</td></tr>"
        for r in records
    )
    return ('<table class="ms-table"><thead><tr><th>Date</th><th>AI suggestion</th><th>Confidence</th>'
            f"<th>Triage indicator</th></tr></thead><tbody>{rows}</tbody></table>")


def _education() -> None:
    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        items = "".join(
            f'<div class="ms-abcde"><div class="letter">{ui.esc(letter)}</div>'
            f'<div class="text"><b>{ui.esc(title)}.</b> {ui.esc(text)}</div></div>'
            for letter, title, text in constants.SKIN_CHECK_GUIDE
        )
        ui.render_html(
            '<div class="ms-card"><div class="ms-eyebrow">Skin self-check guide</div>'
            f"<h3>The ABCDE signs worth noticing</h3>{items}"
            '<p style="margin-top:.6rem">General education only: if any of these apply, '
            "ask a clinician to examine the spot.</p></div>"
        )
    with right:
        for icon, title, text in constants.EDUCATION_TIPS:
            ui.render_html(
                f'<div class="ms-card" style="margin-bottom:.7rem;height:auto"><h4>'
                f'<span class="material-symbols-rounded" style="vertical-align:-6px;color:#0E7C86">'
                f"{ui.esc(icon)}</span> {ui.esc(title)}</h4><p>{ui.esc(text)}</p></div>"
            )


def render() -> None:
    user = ui.current_user()
    try:
        stats = history_service.dashboard_stats(user["id"])
        report_count = len(crud.list_reports(user["id"]))
    except DatabaseError:
        st.error("Your history could not be loaded right now. Please try again.")
        return

    first_name = user["name"].split()[0]
    ui.render_html(
        f'<div class="ms-hero"><h1>Welcome back, {ui.esc(first_name)}</h1>'
        "<p>Upload a clear photo of a skin area to get an AI-assisted, preliminary classification "
        "with educational information and suggested next steps.</p></div>"
    )
    if st.button("Analyze Skin", type="primary", icon=":material/photo_camera:", key="home_cta"):
        ui.go("analyze")

    model_ready, _ = get_model_status()
    latest = stats.latest
    cards = [
        ui.metric_card("Total analyses", str(stats.total), "All time"),
        ui.metric_card("Latest AI suggestion", display_name(latest.condition) if latest else "—",
                       f"{percent(latest.confidence)} model confidence" if latest else "No analyses yet",
                       small=True),
        ui.metric_card("Reports generated", str(report_count), "PDF reports"),
        ui.metric_card("AI model", "Ready" if model_ready else "Not installed",
                       "Prediction available" if model_ready else "See README: model training", small=True),
    ]
    for column, card in zip(st.columns(4, gap="small"), cards):
        column.html(card)

    st.markdown("### Recent analyses")
    if stats.recent:
        ui.render_html(_recent_table(stats.recent))
        if st.button("View full history", icon=":material/history:", key="home_history"):
            ui.go("history")
    else:
        ui.empty_state("No analyses yet", "Your analyses will appear here once you upload your first image.",
                       "photo_camera")

    st.markdown("### Learn")
    _education()
    ui.disclaimer()
