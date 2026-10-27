"""Prediction engine: loads the trained Keras model once and classifies images.

No Streamlit imports here; the UI layer caches ``Predictor.load()`` with
``st.cache_resource`` so the model is loaded once per process.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from config import config, constants
from modules.disease_info import display_name
from prediction import confidence as confidence_rules
from prediction.image_processor import preprocess_image
from utils.logger import get_logger

logger = get_logger("predictor")


class PredictionError(Exception):
    """Base class for prediction failures (messages are safe to show to users)."""


class ModelNotAvailableError(PredictionError):
    """The trained model file is missing or cannot be loaded."""


class ClassNamesError(PredictionError):
    """class_names.json is missing, malformed or inconsistent with the model."""


MODEL_MISSING_HELP = (
    "No trained model was found. Train one with `python -m training.train` "
    "(see README, 'Model training') or place `skin_model.keras` and `class_names.json` "
    "in the `models/` folder."
)


def get_model_status(
    model_path: Path | None = None, class_names_path: Path | None = None
) -> tuple[bool, str]:
    """Cheap file-existence check for the UI (does not load TensorFlow)."""
    model_path = model_path or config.MODEL_PATH
    class_names_path = class_names_path or config.CLASS_NAMES_PATH
    if not Path(model_path).is_file():
        return False, MODEL_MISSING_HELP
    if not Path(class_names_path).is_file():
        return False, "The model file exists but `models/class_names.json` is missing."
    return True, "Model files found."


def load_class_names(path: Path | None = None) -> list[str]:
    path = Path(path or config.CLASS_NAMES_PATH)
    if not path.is_file():
        raise ClassNamesError("Class names file (models/class_names.json) is missing.")
    try:
        names = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ClassNamesError("Class names file is unreadable or not valid JSON.") from None
    if (
        not isinstance(names, list) or len(names) < 2
        or not all(isinstance(n, str) and n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ClassNamesError("Class names file must be a JSON list of at least two unique names.")
    return names


class Predictor:
    def __init__(self, model, class_names: list[str]):
        output_units = model.output_shape[-1]
        if output_units != len(class_names):
            raise ClassNamesError(
                f"Model has {output_units} outputs but class_names.json lists {len(class_names)} classes."
            )
        shape = model.input_shape
        if len(shape) != 4 or shape[1] is None or shape[2] is None:
            raise ModelNotAvailableError("Model must accept fixed-size RGB images (H, W, 3).")
        self.model = model
        self.class_names = class_names
        self.input_size = (int(shape[2]), int(shape[1]))  # (width, height)
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls, model_path: Path | None = None, class_names_path: Path | None = None
    ) -> "Predictor":
        model_path = Path(model_path or config.MODEL_PATH)
        if not model_path.is_file():
            raise ModelNotAvailableError(MODEL_MISSING_HELP)
        class_names = load_class_names(class_names_path)
        try:
            import keras  # imported lazily: TensorFlow import is slow

            model = keras.models.load_model(model_path, compile=False)
        except Exception:
            logger.exception("Failed to load model")
            raise ModelNotAvailableError(
                "The trained model file could not be loaded. It may be corrupted or incompatible."
            ) from None
        logger.info("Model loaded (%d classes, input %s)", len(class_names), model.input_shape[1:3])
        return cls(model, class_names)

    def predict_probabilities(self, image: Image.Image) -> np.ndarray:
        batch = preprocess_image(image, self.input_size)
        try:
            with self._lock:
                output = self.model(batch, training=False)
            probs = np.asarray(output, dtype=np.float64)[0]
        except Exception:
            logger.exception("Model inference failed")
            raise PredictionError("Prediction failed. Please try a different image.") from None
        if (
            probs.shape != (len(self.class_names),) or not np.all(np.isfinite(probs))
            or probs.min() < -1e-6 or abs(probs.sum() - 1.0) > 1e-3
        ):
            logger.error("Model output is not a valid probability distribution")
            raise PredictionError("The model returned an invalid result. Please try again.")
        return np.clip(probs, 0.0, 1.0)

    def predict(self, image: Image.Image) -> dict:
        """Classify one PIL image.

        Returns the required keys (condition, confidence, risk_level, probabilities) plus
        condition_id, confidence_band and top_predictions for the UI/report.
        ``probabilities`` maps display names to the model's raw softmax outputs.
        """
        probs = self.predict_probabilities(image)
        order = np.argsort(probs)[::-1]
        top_index = int(order[0])
        condition_id = self.class_names[top_index]
        confidence = float(probs[top_index])
        probabilities = {display_name(n): float(p) for n, p in zip(self.class_names, probs)}
        top_predictions = [
            {"condition_id": self.class_names[i], "condition": display_name(self.class_names[i]),
             "probability": float(probs[i])}
            for i in order[: constants.TOP_K_PREDICTIONS]
        ]
        return {
            "condition": display_name(condition_id),
            "condition_id": condition_id,
            "confidence": confidence,
            "confidence_band": confidence_rules.confidence_band(confidence),
            "risk_level": confidence_rules.assess_risk(condition_id, confidence),
            "probabilities": probabilities,
            "top_predictions": top_predictions,
        }

    def explain(self, image: Image.Image):
        """Grad-CAM images for ``image`` or None if unavailable for this model."""
        from prediction import explainability

        try:
            batch = preprocess_image(image, self.input_size)
            with self._lock:
                heatmap = explainability.compute_gradcam(self.model, batch)
            base = explainability.resize_for_display(image, self.input_size)
            return explainability.render_heatmap(base, heatmap)
        except explainability.ExplainabilityUnavailable as exc:
            logger.info("Grad-CAM unavailable: %s", exc)
            return None
        except Exception:
            logger.exception("Grad-CAM failed")
            return None
