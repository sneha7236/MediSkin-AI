"""Sign-in and registration page (shown only when no valid session exists)."""
from __future__ import annotations

import streamlit as st

from config import constants
from database.crud import DatabaseError
from modules import authentication as auth
from utils import ui
from utils.validators import ValidationError

UNAVAILABLE = "Service temporarily unavailable. Please try again in a moment."


def _brand() -> None:
    ui.render_html(
        '<div class="ms-auth-brand"><div class="ms-logo-box"><div class="ms-logo-mark">'
        '<span class="material-symbols-rounded">dermatology</span></div>'
        '<div class="ms-logo-text">MediSkin <span>AI</span></div></div>'
        f"<h1>Welcome</h1><p>{ui.esc(constants.APP_TAGLINE)}</p></div>"
    )


def _login_form() -> None:
    with st.form("login_form"):
        email = st.text_input("Email", autocomplete="email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
    if not submitted:
        return
    try:
        user = auth.login_user(email, password)
    except (auth.AuthError, ValidationError) as exc:
        st.error(str(exc))
        return
    auth.start_session(st.session_state, user)
    st.rerun()


def _register_form() -> None:
    with st.form("register_form"):
        name = st.text_input("Full name", autocomplete="name")
        email = st.text_input("Email", autocomplete="email", key="register_email")
        password = st.text_input(
            "Password", type="password", autocomplete="new-password", key="register_password",
            help=f"At least {constants.MIN_PASSWORD_LENGTH} characters with upper- and lower-case letters, "
                 "a digit and a special character.",
        )
        confirm = st.text_input("Confirm password", type="password", autocomplete="new-password")
        submitted = st.form_submit_button("Create account", type="primary", width="stretch")
    if not submitted:
        return
    try:
        user = auth.register_user(name, email, password, confirm)
    except ValidationError as exc:
        st.error(str(exc))
        return
    except DatabaseError:
        st.error(UNAVAILABLE)
        return
    auth.start_session(st.session_state, user)
    st.rerun()


def render() -> None:
    ui.hide_sidebar()
    _, center, _ = st.columns([1, 1.5, 1])
    with center:
        _brand()
        sign_in_tab, register_tab = st.tabs(["Sign in", "Create account"])
        with sign_in_tab:
            _login_form()
        with register_tab:
            _register_form()
        ui.disclaimer()
