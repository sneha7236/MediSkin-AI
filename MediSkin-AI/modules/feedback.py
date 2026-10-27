"""User feedback: rating, comment and whether the prediction was helpful."""
from __future__ import annotations

from config import constants
from database import crud
from database.crud import FeedbackRecord
from utils import security
from utils.logger import get_logger
from utils.validators import ValidationError

logger = get_logger("feedback")


def submit_feedback(user_id: int, rating: int, comment: str = "", helpful: bool | None = None,
                    prediction_id: int | None = None) -> FeedbackRecord:
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValidationError("Please choose a rating from 1 to 5.")
    clean_comment = security.sanitize_text(comment, constants.MAX_COMMENT_LENGTH)
    try:
        record = crud.create_feedback(user_id, rating, clean_comment, helpful, prediction_id)
    except crud.DatabaseError as exc:
        raise ValidationError(str(exc)) from None
    logger.info("Feedback submitted (user=%s, rating=%s)", user_id, rating)
    return record
