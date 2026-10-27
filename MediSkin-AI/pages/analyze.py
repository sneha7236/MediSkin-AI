"""Analyze Skin page: upload -> validate -> preview -> predict -> store -> show results."""
from __future__ import annotations

import streamlit as st

from config import config, constants
from database.crud import DatabaseError
from modules import history as history_service
from prediction.predictor import PredictionError
from utils import ui, validators
from utils.logger import get_logger
from utils.validators import ValidationError

logger = get_logger("analyze")

UPLOAD_CACHE_KEY = "_validated_upload"
UPLOADER_NONCE_KEY = "_uploader_nonce"


def _validated(upload):
    """Validate once per uploaded file (cached by file id). Returns (ValidatedImage | None, error | None)."""
    cached = st.session_state.get(UPLOAD_CACHE_KEY)
    if cached and cached[0] == upload.file_id:
        return cached[1], cached[2]
    try:
        image, error = validators.validate_image_upload(upload.name, upload.getvalue()), None
    except ValidationError as exc:
        image, error = None, str(exc)
    st.session_state[UPLOAD_CACHE_KEY] = (upload.file_id, image, error)
    return image, error


def _guidelines() -> None:
    ui.render_html(
        '<div class="ms-card"><h4>For the best result</h4><ul>'
        "<li>Use natural light and keep the skin area in sharp focus.</li>"
        "<li>Fill most of the frame with the area; avoid filters, shadows and glare.</li>"
        "<li>One area per photo. Remove hair or creams that hide the skin if you can.</li>"
        f"<li>JPG, JPEG or PNG, up to {config.MAX_UPLOAD_MB} MB.</li></ul>"
        '<p style="margin-top:.6rem">Your image is processed in memory and is not saved unless you '
        "generate a PDF report.</p></div>"
    )


def _run_analysis(predictor, image, user: dict) -> bool:
    """Predict, save to history and store the result in the session. True on success."""
    with st.status("Analyzing image…", expanded=True) as status:
        st.write("Image validated.")
        st.write("Preprocessing and running the AI model…")
        try:
            result = predictor.predict(image)
        except PredictionError as exc:
            status.update(label="Analysis failed", state="error")
            st.error(str(exc))
            return False
        except Exception:
            logger.exception("Unexpected prediction failure")
            status.update(label="Analysis failed", state="error")
            st.error("Something went wrong while analyzing the image. Please try again.")
            return False

        st.write("Saving to your history…")
        prediction_id = None
        try:
            prediction_id = history_service.save_prediction(user["id"], result).id
        except DatabaseError:
            st.warning("The result could not be saved to your history, so a PDF report will be unavailable.")
        ui.set_current_result(prediction_id, result, image)
        status.update(label="Analysis complete", state="complete", expanded=False)
    return True


def render() -> None:
    user = ui.current_user()
    ui.page_header("Analyze Skin", "Upload a clear photo to receive a preliminary, AI-assisted classification.",
                   "photo_camera")

    predictor, model_error = ui.get_predictor()
    if predictor is None:
        st.error(f"**AI model unavailable.** {model_error}", icon=":material/error:")

    uploader_key = f"skin_upload_{st.session_state.get(UPLOADER_NONCE_KEY, 0)}"
    left, right = st.columns([1, 1], gap="large")

    with left:
        upload = st.file_uploader(
            "Skin image", type=list(constants.ALLOWED_EXTENSIONS), key=uploader_key,
            help=f"JPG, JPEG or PNG, maximum {config.MAX_UPLOAD_MB} MB.",
        )
        _guidelines()

    validated, error = (_validated(upload) if upload is not None else (None, None))

    with right:
        st.markdown("##### Preview")
        if error:
            st.error(error, icon=":material/error:")
        elif validated is not None:
            width, height = validated.image.size
            st.image(validated.image, width="stretch",
                     caption=f"{upload.name} · {width}×{height}px · {validated.size_bytes / 1024:.0f} KB")
        else:
            ui.empty_state("No image selected", "Choose an image on the left to see a preview here.", "image")

        clicked = st.button("Analyze image", type="primary", icon=":material/biotech:", width="stretch",
                            disabled=predictor is None or validated is None, key="analyze_button")

    if clicked and predictor is not None and validated is not None:
        if _run_analysis(predictor, validated.image, user):
            st.session_state[UPLOADER_NONCE_KEY] = st.session_state.get(UPLOADER_NONCE_KEY, 0) + 1
            st.session_state.pop(UPLOAD_CACHE_KEY, None)
            ui.go("results")

    ui.disclaimer()
