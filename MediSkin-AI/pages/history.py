"""Prediction history: view, download report, delete."""
from __future__ import annotations

import streamlit as st

from config import constants
from database.crud import DatabaseError
from modules import history as history_service
from modules.disease_info import display_name
from modules.report_generator import ReportError
from utils import ui
from utils.helpers import percent

PAGE_SIZE = 10
LIMIT_KEY = "_history_limit"


@st.dialog("Delete this analysis?")
def _confirm_delete(user_id: int, prediction_id: int) -> None:
    st.write("This permanently removes the analysis and its PDF report. This cannot be undone.")
    confirm, cancel = st.columns(2)
    if confirm.button("Delete", type="primary", width="stretch", key=f"confirm_delete_{prediction_id}"):
        try:
            history_service.delete_entry(user_id, prediction_id)
        except DatabaseError:
            st.error("The entry could not be deleted. Please try again.")
            return
        current = ui.get_current_result()
        if current and current["prediction_id"] == prediction_id:
            ui.clear_current_result()
        st.rerun()
    if cancel.button("Cancel", width="stretch", key=f"cancel_delete_{prediction_id}"):
        st.rerun()


def _report_control(user: dict, record, report) -> None:
    key = f"report_{record.id}"
    if report is not None:
        data = history_service.read_report(report)
        if data is not None:
            st.download_button("PDF", data, file_name=f"mediskin_report_{record.id}.pdf", mime="application/pdf",
                               icon=":material/download:", width="stretch", on_click="ignore", key=f"dl_{record.id}")
            return
    if st.button("PDF", icon=":material/picture_as_pdf:", width="stretch", key=key,
                 help="Create a PDF report for this analysis (the image is not stored, so it is omitted)."):
        try:
            history_service.ensure_report(user["id"], user["name"], record.id)
        except (ReportError, DatabaseError) as exc:
            st.error(str(exc) if isinstance(exc, ReportError) else "The report could not be saved.")
            return
        st.rerun()


def _row(user: dict, record, report) -> None:
    with st.container(border=True):
        info, actions = st.columns([3, 2], gap="medium", vertical_alignment="center")
        with info:
            ui.render_html(
                f'<div class="ms-row-label">{ui.esc(display_name(record.condition))}</div>'
                f'<div class="ms-row-sub">{ui.esc(ui.fmt_time(record.created_at))} · '
                f"{ui.esc(percent(record.confidence))} confidence</div>"
                f'<div style="margin-top:.4rem">{ui.risk_badge(record.risk_level)}</div>'
            )
        with actions:
            view, pdf, delete = st.columns(3, gap="small")
            if view.button("View", icon=":material/visibility:", width="stretch", key=f"view_{record.id}"):
                ui.set_current_result(record.id, history_service.result_from_record(record))
                ui.go("results")
            with pdf:
                _report_control(user, record, report)
            if delete.button("Delete", icon=":material/delete:", width="stretch", key=f"delete_{record.id}"):
                _confirm_delete(user["id"], record.id)


def render() -> None:
    user = ui.current_user()
    ui.page_header("Prediction History", "Review, download or delete your earlier analyses.", "history")

    try:
        records = history_service.list_history(user["id"])
        reports = history_service.reports_by_prediction(user["id"])
    except DatabaseError:
        st.error("Your history could not be loaded right now. Please try again.")
        return

    if not records:
        ui.empty_state("No history yet", "Analyses you run will be listed here.", "history")
        if st.button("Analyze Skin", type="primary", icon=":material/photo_camera:", key="history_to_analyze"):
            ui.go("analyze")
        ui.disclaimer()
        return

    conditions = sorted({display_name(r.condition) for r in records})
    chosen = st.multiselect("Filter by AI suggestion", conditions, placeholder="All conditions")
    shown = [r for r in records if not chosen or display_name(r.condition) in chosen]
    noun = "analysis" if len(records) == 1 else "analyses"
    st.caption(f"{len(shown)} of {len(records)} {noun}")

    limit = st.session_state.get(LIMIT_KEY, PAGE_SIZE)
    for record in shown[:limit]:
        _row(user, record, reports.get(record.id))
    if len(shown) > limit and st.button("Show more", key="history_more"):
        st.session_state[LIMIT_KEY] = min(limit + PAGE_SIZE, constants.HISTORY_PAGE_SIZE * 4)
        st.rerun()
    ui.disclaimer()
