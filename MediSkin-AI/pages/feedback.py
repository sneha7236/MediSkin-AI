"""Feedback page: rating, comment and whether a specific analysis was helpful."""
from __future__ import annotations

import streamlit as st

from config import constants
from database import crud
from database.crud import DatabaseError
from modules import feedback as feedback_service
from modules import history as history_service
from modules.disease_info import display_name
from utils import ui
from utils.validators import ValidationError

GENERAL = "General feedback (not about a specific analysis)"
HELPFUL_OPTIONS = {"Helpful": True, "Not helpful": False, "Prefer not to say": None}


def render() -> None:
    user = ui.current_user()
    ui.page_header("Feedback", "Tell us how useful the application and its results were.", "rate_review")

    try:
        records = history_service.list_history(user["id"])[:25]
    except DatabaseError:
        records = []
    labels = {r.id: f"{ui.fmt_time(r.created_at)} · {display_name(r.condition)}" for r in records}
    options = [None, *labels]
    current = ui.get_current_result()
    preselect = current["prediction_id"] if current and current["prediction_id"] in labels else None

    with st.form("feedback_form", clear_on_submit=True):
        prediction_id = st.selectbox("About which analysis?", options, index=options.index(preselect),
                                     format_func=lambda pid: GENERAL if pid is None else labels[pid])
        helpful_label = st.radio("Was the result helpful?", list(HELPFUL_OPTIONS), index=2, horizontal=True)
        stars = st.feedback("stars")
        st.caption("Rating: select 1 to 5 stars.")
        comment = st.text_area("Comments (optional)", max_chars=constants.MAX_COMMENT_LENGTH,
                               placeholder="What worked well? What could be better?")
        submitted = st.form_submit_button("Submit feedback", type="primary", icon=":material/send:")

    if submitted:
        try:
            feedback_service.submit_feedback(
                user["id"], (stars + 1) if stars is not None else 0, comment,
                HELPFUL_OPTIONS[helpful_label], prediction_id,
            )
        except ValidationError as exc:
            st.error(str(exc))
        except DatabaseError:
            st.error("Your feedback could not be saved right now. Please try again.")
        else:
            st.success("Thank you! Your feedback has been submitted.", icon=":material/check_circle:")

    try:
        previous = crud.list_feedback(user["id"])
    except DatabaseError:
        previous = []
    if previous:
        st.markdown("### Your previous feedback")
        for item in previous[:5]:
            about = labels.get(item.prediction_id, "General feedback")
            with st.container(border=True):
                st.html(
                    f'<div class="ms-row-label">{"★" * item.rating}{"☆" * (5 - item.rating)}</div>'
                    f'<div class="ms-row-sub">{ui.esc(ui.fmt_time(item.created_at))} · {ui.esc(about)}</div>'
                    + (f"<p style='margin:.4rem 0 0'>{ui.esc(item.comment)}</p>" if item.comment else "")
                )
    ui.disclaimer()
