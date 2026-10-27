"""About page: purpose, how it works, how to read results, privacy and limitations."""
from __future__ import annotations

import streamlit as st

from config import config, constants
from modules import analytics
from utils import ui

STEPS = (
    ("1", "Upload", "Choose a JPG, JPEG or PNG photo. It is validated for type, size and integrity."),
    ("2", "Preprocess", "The image is oriented, resized to the model input size and normalised."),
    ("3", "Classify", "A trained deep-learning model returns a probability for each supported category."),
    ("4", "Interpret", "The top category, its probability and an informational triage indicator are shown."),
    ("5", "Act", "Read general guidance, find a clinic, and download a PDF to share with a professional."),
)
LIMITATIONS = (
    "The model only knows the categories it was trained on and can be confidently wrong.",
    "Photo quality, lighting, skin tone representation in the training data and image artefacts affect results.",
    "It cannot assess symptoms, history or how a lesion has changed over time.",
    "Triage indicators are informational and are not clinical risk scores.",
    "A normal-looking result does not rule out a skin condition. Seek care for anything that worries you.",
)


def render() -> None:
    ui.page_header("About MediSkin AI", constants.APP_TAGLINE, "info")
    ui.render_html(
        '<div class="ms-card"><h3>What this is</h3><p>MediSkin AI is an academic, AI-assisted screening and '
        "education tool. It gives a preliminary classification of a visible skin condition from a photo, "
        "explains what that category generally means, and points you to professional care. "
        "It does not diagnose, treat or replace a clinician.</p></div>"
    )

    st.markdown("### How it works")
    steps = "".join(
        f'<div class="ms-abcde"><div class="letter">{n}</div><div class="text"><b>{ui.esc(title)}.</b> '
        f"{ui.esc(text)}</div></div>" for n, title, text in STEPS
    )
    ui.render_html(f'<div class="ms-card">{steps}</div>')

    left, right = st.columns(2, gap="medium")
    with left:
        ui.render_html(
            '<div class="ms-card"><h4>Reading the confidence level</h4><ul>'
            f"<li><b>High</b>: {constants.HIGH_CONFIDENCE:.2f} or above</li>"
            f"<li><b>Moderate</b>: {constants.MODERATE_CONFIDENCE:.2f} to below {constants.HIGH_CONFIDENCE:.2f}</li>"
            f"<li><b>Low</b>: below {constants.MODERATE_CONFIDENCE:.2f} (the triage shows “Uncertain”)</li></ul>"
            "<p style='margin-top:.5rem'>Confidence is the model's own probability, not medical certainty.</p></div>"
        )
    with right:
        ui.render_html(
            '<div class="ms-card"><h4>Triage indicators</h4><ul>'
            f"<li>{ui.risk_badge(constants.RISK_ROUTINE)} category is commonly harmless</li>"
            f"<li>{ui.risk_badge(constants.RISK_MONITOR)} worth a professional look</li>"
            f"<li>{ui.risk_badge(constants.RISK_PRIORITY)} clinicians usually examine promptly</li>"
            f"<li>{ui.risk_badge(constants.RISK_UNCERTAIN)} model unsure; get an expert opinion</li></ul></div>"
        )

    left, right = st.columns(2, gap="medium")
    with left:
        items = "".join(f"<li>{ui.esc(x)}</li>" for x in LIMITATIONS)
        ui.render_html(f'<div class="ms-card"><h4>Limitations</h4><ul>{items}</ul></div>')
    with right:
        metrics = analytics.load_metrics()
        model = (metrics or {}).get("model") or {}
        model_line = (f"Model: {ui.esc(model.get('architecture', 'trained classifier'))}, "
                      f"{len(metrics['class_names'])} categories." if metrics else
                      "Model details appear here after a model has been trained and evaluated.")
        ui.render_html(
            '<div class="ms-card"><h4>Privacy and data</h4><ul>'
            "<li>Uploaded images are analysed in memory and are not saved after analysis.</li>"
            "<li>If you generate a PDF report, the image is embedded in that report file.</li>"
            "<li>Passwords are stored only as salted hashes.</li>"
            f"<li>{model_line}</li></ul></div>"
        )
    st.caption(f"{constants.APP_NAME} v{constants.APP_VERSION} · "
               f"max upload {config.MAX_UPLOAD_MB} MB · Streamlit, TensorFlow/Keras, SQLite, ReportLab")
    ui.disclaimer()
