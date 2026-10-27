"""Transparent confidence bands and informational triage categories.

Rules (applied in this order):
1. Confidence band comes only from the model's top probability:
   High >= 0.75, Moderate 0.50-0.74, Low < 0.50.
2. Low confidence  -> triage "Uncertain" (the model is not sure; get a professional opinion).
3. Otherwise       -> the base triage level of the predicted class
   (see config.constants.CLASS_BASE_RISK).

Confidence is the model's probability output, NOT medical certainty. Triage is an
informational indicator, NOT a clinical diagnosis.
"""
from __future__ import annotations

from config import constants as c


def confidence_band(confidence: float) -> str:
    if confidence >= c.HIGH_CONFIDENCE:
        return c.CONFIDENCE_HIGH
    if confidence >= c.MODERATE_CONFIDENCE:
        return c.CONFIDENCE_MODERATE
    return c.CONFIDENCE_LOW


def assess_risk(condition_id: str, confidence: float) -> str:
    if confidence_band(confidence) == c.CONFIDENCE_LOW:
        return c.RISK_UNCERTAIN
    return c.CLASS_BASE_RISK.get(condition_id, c.RISK_MONITOR)


def next_steps_guidance(risk_level: str, band: str) -> str:
    """General (non-personalised) next-step text for the results page and PDF."""
    if risk_level == c.RISK_UNCERTAIN or band == c.CONFIDENCE_LOW:
        return (
            "The model is not confident about this image. This does not mean the skin is fine or "
            "that something is wrong. A clinician's examination is the right way to find out. "
            "Consider retaking a clear, well-lit, in-focus photo, and consult a dermatologist."
        )
    if risk_level == c.RISK_PRIORITY:
        return (
            "This category is one that clinicians usually want to examine promptly. Please arrange "
            "an appointment with a dermatologist soon, and do not attempt to treat the area yourself."
        )
    if risk_level == c.RISK_MONITOR:
        return (
            "Consider having this checked by a healthcare professional, particularly if it is new, "
            "changing, painful or bleeding. Keep track of its appearance in the meantime."
        )
    return (
        "This category is commonly harmless, but an automated image check cannot confirm that. "
        "Monitor the area for changes and consult a dermatologist if you are unsure or notice any "
        "change in size, shape, colour, or new symptoms."
    )


def confidence_explanation(confidence: float) -> str:
    band = confidence_band(confidence)
    return (
        f"{band} model confidence ({confidence * 100:.1f}%). This is the model's own probability "
        "estimate for the top category, not a measure of medical certainty."
    )
