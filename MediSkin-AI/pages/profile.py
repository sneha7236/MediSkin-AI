"""User profile: account details, name update and password change."""
from __future__ import annotations

import streamlit as st

from config import constants
from database import crud
from database.crud import DatabaseError
from modules import authentication as auth
from utils import ui
from utils.validators import ValidationError


def _update_name(user: dict) -> None:
    with st.form("name_form"):
        name = st.text_input("Full name", value=user["name"])
        submitted = st.form_submit_button("Save name", type="primary")
    if not submitted:
        return
    try:
        updated = auth.update_profile_name(user["id"], name)
    except (auth.AuthError, ValidationError) as exc:
        st.error(str(exc))
    except DatabaseError:
        st.error("Your profile could not be updated right now. Please try again.")
    else:
        st.session_state[auth.SESSION_USER_KEY]["name"] = updated.name
        st.success("Name updated.")
        st.rerun()


def _change_password(user: dict) -> None:
    with st.form("password_form", clear_on_submit=True):
        current = st.text_input("Current password", type="password", autocomplete="current-password")
        new = st.text_input("New password", type="password", autocomplete="new-password",
                            help=f"At least {constants.MIN_PASSWORD_LENGTH} characters with upper- and lower-case "
                                 "letters, a digit and a special character.")
        confirm = st.text_input("Confirm new password", type="password", autocomplete="new-password")
        submitted = st.form_submit_button("Change password", type="primary")
    if not submitted:
        return
    try:
        auth.change_password(user["id"], current, new, confirm)
    except (auth.AuthError, ValidationError) as exc:
        st.error(str(exc))
    except DatabaseError:
        st.error("Your password could not be changed right now. Please try again.")
    else:
        st.success("Password changed.")


def render() -> None:
    user = ui.current_user()
    ui.page_header("Profile", "Manage your account.", "account_circle")

    try:
        record = crud.get_user(user["id"])
        total = crud.count_predictions(user["id"])
        reports = len(crud.list_reports(user["id"]))
    except DatabaseError:
        st.error("Your profile could not be loaded right now. Please try again.")
        return

    cards = [
        ui.metric_card("Name", user["name"], small=True),
        ui.metric_card("Email", user["email"], small=True),
        ui.metric_card("Member since", record.created_at.strftime("%d %b %Y") if record else "—", small=True),
        ui.metric_card("Analyses / reports", f"{total} / {reports}", small=True),
    ]
    for column, card in zip(st.columns(4, gap="small"), cards):
        column.html(card)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Update name")
        _update_name(user)
    with right:
        st.markdown("### Change password")
        _change_password(user)
    ui.disclaimer()
