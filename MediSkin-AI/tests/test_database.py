import pytest

from database import crud


def _user(email="a@example.com"):
    return crud.create_user("Alice", email, "hash")


def test_create_and_get_user(db):
    user = _user()
    assert user.id and crud.get_user(user.id).email == "a@example.com"
    creds = crud.get_user_credentials_by_email("a@example.com")
    assert creds.password_hash == "hash"
    assert crud.get_user_credentials_by_email("none@example.com") is None


def test_duplicate_email_rejected(db):
    _user()
    with pytest.raises(crud.DuplicateEmailError):
        _user()


def test_prediction_roundtrip_and_count(db):
    user = _user()
    pred = crud.create_prediction(user.id, "mel", 0.82, "Higher concern", {"mel": 0.82, "nv": 0.18})
    assert crud.count_predictions(user.id) == 1
    fetched = crud.get_prediction(user.id, pred.id)
    assert fetched.probabilities["mel"] == pytest.approx(0.82)
    assert crud.list_predictions(user.id, limit=5)[0].id == pred.id


def test_predictions_are_user_scoped(db):
    u1, u2 = _user(), _user("b@example.com")
    pred = crud.create_prediction(u1.id, "nv", 0.9, "Lower concern", {})
    assert crud.get_prediction(u2.id, pred.id) is None
    assert crud.delete_prediction(u2.id, pred.id) == []
    assert crud.count_predictions(u1.id) == 1
    with pytest.raises(crud.DatabaseError):
        crud.create_report(u2.id, pred.id, "x.pdf")


def test_delete_prediction_cascades_and_returns_report_paths(db):
    user = _user()
    pred = crud.create_prediction(user.id, "nv", 0.9, "Lower concern", {})
    crud.create_report(user.id, pred.id, "/tmp/r.pdf")
    crud.create_feedback(user.id, 4, "ok", True, pred.id)
    assert crud.delete_prediction(user.id, pred.id) == ["/tmp/r.pdf"]
    assert crud.list_reports(user.id) == []
    assert crud.list_feedback(user.id) == []


def test_feedback_general(db):
    user = _user()
    fb = crud.create_feedback(user.id, 5, "great", None, None)
    assert fb.rating == 5 and crud.list_feedback(user.id)[0].comment == "great"
