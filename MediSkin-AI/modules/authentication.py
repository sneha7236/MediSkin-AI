"""Registration, login and session handling (UI-framework independent).

Session functions operate on any dict-like object, so Streamlit's ``st.session_state``
can be passed in directly while tests use a plain dict.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

from config import config
from database import crud
from database.crud import DatabaseError, DuplicateEmailError, UserRecord
from utils import security, validators
from utils.helpers import utcnow
from utils.logger import get_logger
from utils.validators import ValidationError

logger = get_logger("auth")

SESSION_USER_KEY = "auth_user"
SESSION_STARTED_KEY = "auth_last_active"

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300
_failed_attempts: dict[str, deque[float]] = defaultdict(deque)

# Verifying against this when the email is unknown keeps response time similar,
# so login timing does not reveal which emails are registered.
_DUMMY_HASH = security.hash_password("dummy-password-for-timing")

GENERIC_LOGIN_ERROR = "Invalid email or password."


class AuthError(ValueError):
    """Authentication failure with a user-friendly message."""


# --- Registration / login ------------------------------------------------
def register_user(name: str, email: str, password: str, confirm_password: str) -> UserRecord:
    clean_name = validators.validate_name(name)
    clean_email = validators.validate_email(email)
    validators.validate_password_strength(password)
    if password != confirm_password:
        raise ValidationError("Passwords do not match.")
    try:
        user = crud.create_user(clean_name, clean_email, security.hash_password(password))
    except DuplicateEmailError:
        raise ValidationError("An account with this email already exists.") from None
    logger.info("User registered (id=%s)", user.id)
    return user


def _prune(attempts: deque[float], now: float) -> None:
    while attempts and now - attempts[0] > LOCKOUT_SECONDS:
        attempts.popleft()


def _is_locked(email: str) -> bool:
    attempts = _failed_attempts[email]
    _prune(attempts, time.monotonic())
    return len(attempts) >= MAX_FAILED_ATTEMPTS


def reset_login_throttle() -> None:
    _failed_attempts.clear()


def login_user(email: str, password: str) -> UserRecord:
    if not (email or "").strip() or not password:
        raise AuthError("Email and password are required.")
    key = email.strip().lower()
    if _is_locked(key):
        logger.warning("Login blocked by throttle (id unknown)")
        raise AuthError("Too many failed attempts. Please wait a few minutes and try again.")

    try:
        creds = crud.get_user_credentials_by_email(key)
    except DatabaseError:
        raise AuthError("Service temporarily unavailable. Please try again.") from None

    hash_to_check = creds.password_hash if creds else _DUMMY_HASH
    password_ok = security.verify_password(password, hash_to_check)
    if creds is None or not password_ok:
        _failed_attempts[key].append(time.monotonic())
        logger.info("Failed login attempt")
        raise AuthError(GENERIC_LOGIN_ERROR)

    _failed_attempts.pop(key, None)
    logger.info("User logged in (id=%s)", creds.id)
    return UserRecord(creds.id, creds.name, creds.email, creds.created_at)


def change_password(user_id: int, current_password: str, new_password: str, confirm: str) -> None:
    creds = crud.get_user_credentials_by_id(user_id)
    if creds is None or not security.verify_password(current_password, creds.password_hash):
        raise AuthError("Current password is incorrect.")
    validators.validate_password_strength(new_password)
    if new_password != confirm:
        raise ValidationError("New passwords do not match.")
    crud.update_password_hash(user_id, security.hash_password(new_password))
    logger.info("Password changed (id=%s)", user_id)


def update_profile_name(user_id: int, name: str) -> UserRecord:
    clean = validators.validate_name(name)
    user = crud.update_user_name(user_id, clean)
    if user is None:
        raise AuthError("Account not found.")
    return user


# --- Session handling ----------------------------------------------------
def start_session(state, user: UserRecord) -> None:
    state[SESSION_USER_KEY] = {"id": user.id, "name": user.name, "email": user.email}
    state[SESSION_STARTED_KEY] = utcnow()


def end_session(state) -> None:
    user = state.get(SESSION_USER_KEY)
    for key in list(state.keys()):
        del state[key]
    if user:
        logger.info("User logged out (id=%s)", user["id"])


def get_current_user(state) -> dict | None:
    """Return the logged-in user dict, or None if absent/expired. Refreshes activity time."""
    user = state.get(SESSION_USER_KEY)
    last_active: datetime | None = state.get(SESSION_STARTED_KEY)
    if not user or last_active is None:
        return None
    if utcnow() - last_active > timedelta(minutes=config.SESSION_TIMEOUT_MINUTES):
        logger.info("Session expired (id=%s)", user["id"])
        end_session(state)
        return None
    state[SESSION_STARTED_KEY] = utcnow()
    return user
