"""Results page: prediction card, explanation, guidance, report download and next actions."""
from __future__ import annotations

import streamlit as st

from config import constants
from database.crud import DatabaseError
from modules import history as history_service
from modules.disease_info import get_disease_info
from modules.report_generator import ReportError
from prediction import confidence as confidence_rules
from utils import ui
from utils.helpers import percent
from utils.logger import get_logger

logger = get_logger("results")


def _bullets(items) -> str:
    return "<ul>" + "".join(f"<li>{ui.esc(item)}</li>" for item in items) + "</ul>"


def _summary(result: dict) -> str:
    confidence = result["confidence"]
    return (
        '<div class="ms-eyebrow">AI suggestion (not a diagnosis)</div>'
        f'<div class="ms-result-title">{ui.esc(result["condition"])}</div>'
        f'<div class="ms-conf"><span class="pct">{ui.esc(percent(confidence))}</span>'
        '<span class="ms-eyebrow">model confidence</span></div>'
        f'<div class="ms-badges">{ui.band_badge(result["confidence_band"])}'
        f'{ui.risk_badge(result["risk_level"])}</div>'
        f'<p style="color:#475569;font-size:.9rem;margin:0">'
        f'{ui.esc(confidence_rules.confidence_explanation(confidence))}</p>'
    )


def _top_predictions(result: dict) -> None:
    items = [(p["condition"], p["probability"]) for p in result["top_predictions"]]
    ui.render_html(f'<div class="ms-card"><h4>Top predictions</h4>{ui.probability_bars(items)}</div>')


def _gradcam(state: dict) -> None:
    """Attention visualization; computed once per result and hidden if unavailable."""
    image = state["image"]
    if image is None:
        return
    if state["gradcam"] == "pending":
        predictor, _ = ui.get_predictor()
        state["gradcam"] = (predictor.explain(image) if predictor is not None else None) or False
    if not state["gradcam"]:
        return
    original, heatmap, overlay = state["gradcam"]
    st.markdown("### AI attention visualization")
    ui.render_html(f'<div class="ms-note-ai">{ui.esc(constants.GRADCAM_DISCLAIMER)}</div>')
    for column, (label, picture) in zip(st.columns(3, gap="small"),
                                        (("Original", original), ("Heatmap", heatmap), ("Overlay", overlay))):
        column.image(picture, caption=label, width="stretch")


def _about(info) -> None:
    st.markdown("### About this condition")
    ui.render_html(
        f'<div class="ms-card"><h3>{ui.esc(info.name)}</h3><p>{ui.esc(info.description)}</p></div>'
    )
    left, right = st.columns(2, gap="medium")
    left.html(f'<div class="ms-card"><h4>Common signs</h4>{_bullets(info.common_signs)}</div>')
    right.html(f'<div class="ms-card"><h4>General risk factors</h4>{_bullets(info.risk_factors)}</div>')


def _next_steps(result: dict, info) -> None:
    st.markdown("### What to do next")
    guidance = confidence_rules.next_steps_guidance(result["risk_level"], result["confidence_band"])
    ui.render_html(f'<div class="ms-card"><h4>Suggested next step</h4><p>{ui.esc(guidance)}</p></div>')
    left, right = st.columns(2, gap="medium")
    left.html(f'<div class="ms-card"><h4>General care and prevention</h4>{_bullets(info.care_and_prevention)}</div>')
    right.html(f'<div class="ms-card"><h4>When to seek professional advice</h4>{_bullets(info.seek_care_when)}</div>')


def _report_button(user: dict, state: dict) -> None:
    prediction_id = state["prediction_id"]
    if prediction_id is None:
        st.button("Download PDF report", icon=":material/picture_as_pdf:", disabled=True, width="stretch",
                  help="This result was not saved, so a report cannot be created.")
        return
    existing = history_service.get_report_for_prediction(user["id"], prediction_id)
    if existing is None:
        if st.button("Generate PDF report", icon=":material/picture_as_pdf:", width="stretch",
                     key="generate_report"):
            try:
                with st.spinner("Creating your report…"):
                    history_service.ensure_report(user["id"], user["name"], prediction_id, state["image"])
            except (ReportError, DatabaseError) as exc:
                st.error(str(exc) if isinstance(exc, ReportError) else "The report could not be saved.")
                return
            st.rerun()
        return
    data = history_service.read_report(existing)
    if data is None:
        st.error("The report file is no longer available.")
        return
    st.download_button("Download PDF report", data, file_name=f"mediskin_report_{prediction_id}.pdf",
                       mime="application/pdf", icon=":material/download:", type="primary", width="stretch",
                       on_click="ignore", key="download_report")


def _actions(user: dict, state: dict) -> None:
    st.markdown("### Take action")
    first, second, third, fourth = st.columns(4, gap="small")
    with first:
        if st.button("Find a dermatologist", icon=":material/local_hospital:", width="stretch", key="to_hospitals"):
            ui.go("hospitals")
    with second:
        _report_button(user, state)
    with third:
        if st.button("Rate this result", icon=":material/rate_review:", width="stretch", key="to_feedback"):
            ui.go("feedback")
    with fourth:
        if st.button("Analyze another image", icon=":material/add_a_photo:", width="stretch", key="to_analyze"):
            ui.clear_current_result()
            ui.go("analyze")


def render() -> None:
    user = ui.current_user()
    ui.page_header("AI Analysis Result", "A preliminary result for information only.", "clinical_notes")

    state = ui.get_current_result()
    if not state:
        ui.empty_state("No result to show", "Analyze an image, or open an earlier analysis from your history.",
                       "clinical_notes")
        if st.button("Analyze Skin", type="primary", icon=":material/photo_camera:", key="empty_to_analyze"):
            ui.go("analyze")
        ui.disclaimer()
        return

    result = state["result"]
    info = get_disease_info(result["condition_id"])

    image_col, summary_col = st.columns([1, 1.5], gap="large")
    with image_col:
        if state["image"] is not None:
            st.image(state["image"], caption="Analyzed image", width="stretch")
        else:
            ui.empty_state("Image not stored", "Uploaded images are not kept after analysis, so this stored "
                           "result has no picture.", "hide_image")
    with summary_col:
        ui.render_html(f'<div class="ms-card">{_summary(result)}</div>')
    _top_predictions(result)

    _gradcam(state)
    _about(info)
    _next_steps(result, info)
    _actions(user, state)
    ui.disclaimer()
