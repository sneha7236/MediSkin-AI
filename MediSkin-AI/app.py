"""MediSkin AI — Streamlit entry point.

Responsibilities: page configuration, one-time startup, session guard and navigation.
All features live in ``pages/`` (UI) and ``modules/``, ``prediction/``, ``database/`` (logic).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="MediSkin AI",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="auto",
)

from config import config, constants  # noqa: E402
from database import database  # noqa: E402
from modules import authentication as auth  # noqa: E402
from pages import (about, analytics, analyze, auth as auth_page, feedback, history, home,  # noqa: E402
                   hospitals, profile, reports, results)
from utils import ui  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("app")

ASSETS = config.BASE_DIR / "assets" / "images"


@st.cache_resource(show_spinner=False)
def _startup() -> bool:
    """Runs once per server process."""
    config.ensure_runtime_dirs()
    database.init_engine()
    logger.info("Application startup (%s v%s)", constants.APP_NAME, constants.APP_VERSION)
    return True


def _build_navigation() -> tuple[dict, dict]:
    """Return (flat page registry for ui.go, grouped navigation structure)."""
    def page(key, func, title, icon, **kwargs):
        return key, st.Page(func, title=title, icon=f":material/{icon}:", url_path=key, **kwargs)

    groups = {
        "Workspace": [
            page("home", home.render, "Home", "dashboard", default=True),
            page("analyze", analyze.render, "Analyze Skin", "photo_camera"),
            page("results", results.render, "Results", "clinical_notes"),
        ],
        "Records": [
            page("history", history.render, "Prediction History", "history"),
            page("reports", reports.render, "Reports", "description"),
        ],
        "Care & Insights": [
            page("hospitals", hospitals.render, "Hospitals", "local_hospital"),
            page("analytics", analytics.render, "Model Analytics", "monitoring"),
        ],
        "General": [
            page("feedback", feedback.render, "Feedback", "rate_review"),
            page("profile", profile.render, "Profile", "account_circle"),
            page("about", about.render, "About", "info"),
        ],
    }
    registry = {key: pg for items in groups.values() for key, pg in items}
    return registry, {name: [pg for _, pg in items] for name, items in groups.items()}


def _login_navigation():
    """Logged-out routing: every known path shows the sign-in form.

    Without this, deep-linking to e.g. /history while signed out would trigger Streamlit's
    "page not found" dialog; with it the user signs in and lands on the page they asked for.
    """
    def stub(key: str):
        def render_login() -> None:
            auth_page.render()
        render_login.__name__ = f"login_for_{key}"
        return render_login

    registry, _ = _build_navigation()
    pages = [st.Page(auth_page.render, title="Sign in", url_path="login", default=True)]
    pages += [st.Page(stub(key), title=key.title(), url_path=key) for key in registry if key != "home"]
    pages.append(st.Page(stub("home"), title="Home", url_path="home"))
    return st.navigation(pages, position="hidden")


def _render_sidebar(user: dict) -> None:
    with st.sidebar:
        st.divider()
        ui.render_html(
            f'<div class="ms-side-user"><div class="name">{ui.esc(user["name"])}</div>'
            f'<div class="email">{ui.esc(user["email"])}</div></div>'
        )
        if st.button("Sign out", icon=":material/logout:", width="stretch", key="logout_button"):
            auth.end_session(st.session_state)
            st.rerun()
        ui.render_html(
            '<div class="ms-side-note">AI-assisted screening for information only. '
            "Not a medical diagnosis; consult a qualified healthcare professional.</div>"
        )


def main() -> None:
    ui.load_css()
    try:
        _startup()
    except Exception:
        logger.exception("Startup failed")
        st.error("The application could not start. Please check the configuration and try again.")
        st.stop()

    user = auth.get_current_user(st.session_state)
    if user is None:
        navigation = _login_navigation()
    else:
        st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo-icon.svg"), size="large")
        registry, grouped = _build_navigation()
        ui.register_pages(registry)
        navigation = st.navigation(grouped)
        _render_sidebar(user)

    try:
        navigation.run()
    except Exception:
        logger.exception("Unhandled error while rendering a page")
        st.error("Something went wrong while loading this page. Please try again.")


main()
