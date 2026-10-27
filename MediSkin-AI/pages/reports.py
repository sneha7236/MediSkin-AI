"""Report history: every PDF report the user has generated."""
from __future__ import annotations

import streamlit as st

from database import crud
from database.crud import DatabaseError
from modules import history as history_service
from modules.disease_info import display_name
from utils import ui
from utils.helpers import percent


def render() -> None:
    user = ui.current_user()
    ui.page_header("Reports", "Download the PDF reports you have generated.", "description")

    try:
        reports = crud.list_reports(user["id"])
        predictions = {p.id: p for p in history_service.list_history(user["id"])}
    except DatabaseError:
        st.error("Your reports could not be loaded right now. Please try again.")
        return

    if not reports:
        ui.empty_state("No reports yet", "Generate a PDF from the Results page or from your history.",
                       "description")
        ui.disclaimer()
        return

    st.caption(f"{len(reports)} report(s)")
    for report in reports:
        prediction = predictions.get(report.prediction_id)
        with st.container(border=True):
            info, action = st.columns([4, 1.4], vertical_alignment="center")
            title = display_name(prediction.condition) if prediction else "Analysis report"
            detail = f" · {percent(prediction.confidence)} confidence" if prediction else ""
            info.html(
                f'<div class="ms-row-label">{ui.esc(title)}</div>'
                f'<div class="ms-row-sub">Created {ui.esc(ui.fmt_time(report.created_at))}{ui.esc(detail)}</div>'
            )
            data = history_service.read_report(report)
            if data is None:
                action.caption("File unavailable")
            else:
                action.download_button("Download", data, file_name=f"mediskin_report_{report.id}.pdf",
                                       mime="application/pdf", icon=":material/download:", width="stretch",
                                       on_click="ignore", key=f"report_dl_{report.id}")
    ui.disclaimer()
