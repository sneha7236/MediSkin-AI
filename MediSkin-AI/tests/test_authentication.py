from datetime import timedelta

import pytest

from database import crud
from modules import authentication as auth
from utils.helpers import utcnow
from utils.validators import ValidationError

PW = "Str0ng!Pass"


@pytest.fixture(autouse=True)
def _reset_throttle():
    auth.reset_login_throttle()


def _register(email="jane@example.com"):
    return auth.register_user("Jane Doe", email, PW, PW)


def test_register_stores_hash_not_plaintext(db):
    user = _register()
    creds = crud.get_user_credentials_by_email("jane@example.com")
    assert creds.id == user.id
    assert creds.password_hash != PW and creds.password_hash.startswith("$2")


def test_register_validation(db):
    with pytest.raises(ValidationError):
        auth.register_user("", "a@b.com", PW, PW)
    with pytest.raises(ValidationError):
        auth.register_user("Jane", "bad-email", PW, PW)
    with pytest.raises(ValidationError):
        auth.register_user("Jane", "a@b.com", "weak", "weak")
    with pytest.raises(ValidationError, match="do not match"):
        auth.register_user("Jane", "a@b.com", PW, PW + "x")


def test_duplicate_email_case_insensitive(db):
    _register("jane@example.com")
    with pytest.raises(ValidationError, match="already exists"):
        _register("JANE@example.com")


def test_login_success_and_failures(db):
    _register()
    assert auth.login_user(" Jane@Example.com ", PW).name == "Jane Doe"
    with pytest.raises(auth.AuthError) as wrong_pw:
        auth.login_user("jane@example.com", "Wrong!Pass1")
    with pytest.raises(auth.AuthError) as unknown:
        auth.login_user("ghost@example.com", PW)
    assert str(wrong_pw.value) == str(unknown.value) == auth.GENERIC_LOGIN_ERROR
    with pytest.raises(auth.AuthError):
        auth.login_user("", "")


def test_login_throttle(db):
    _register()
    for _ in range(auth.MAX_FAILED_ATTEMPTS):
        with pytest.raises(auth.AuthError):
            auth.login_user("jane@example.com", "bad")
    with pytest.raises(auth.AuthError, match="Too many"):
        auth.login_user("jane@example.com", PW)  # even correct password is blocked


def test_session_lifecycle_and_timeout(db):
    user = _register()
    state: dict = {"other": 1}
    assert auth.get_current_user(state) is None
    auth.start_session(state, user)
    assert auth.get_current_user(state)["email"] == "jane@example.com"
    assert "password_hash" not in str(state)

    state[auth.SESSION_STARTED_KEY] = utcnow() - timedelta(days=1)
    assert auth.get_current_user(state) is None
    assert state == {}

    auth.start_session(state, user)
    auth.end_session(state)
    assert auth.get_current_user(state) is None


def test_change_password(db):
    user = _register()
    with pytest.raises(auth.AuthError):
        auth.change_password(user.id, "wrong", "N3w!Passw0rd", "N3w!Passw0rd")
    with pytest.raises(ValidationError):
        auth.change_password(user.id, PW, "weak", "weak")
    auth.change_password(user.id, PW, "N3w!Passw0rd", "N3w!Passw0rd")
    assert auth.login_user("jane@example.com", "N3w!Passw0rd").id == user.id


def test_update_profile_name(db):
    user = _register()
    assert auth.update_profile_name(user.id, "  Jane  Q ").name == "Jane Q"
    with pytest.raises(ValidationError):
        auth.update_profile_name(user.id, "")
