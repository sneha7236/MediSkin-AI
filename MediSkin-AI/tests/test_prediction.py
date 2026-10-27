"""Prediction engine, confidence rules, preprocessing and the training pipeline.

The models used here are tiny, randomly-initialised or briefly trained on synthetic images
inside a temp directory. They exist only to exercise the code paths; they are never copied
to ``models/`` and no accuracy from them is ever displayed by the application.
"""
import json

import numpy as np
import pytest
from PIL import Image

from config import config, constants
from prediction import confidence
from prediction.image_processor import preprocess_image
from prediction.predictor import (
    ClassNamesError, ModelNotAvailableError, PredictionError, Predictor, get_model_status,
    load_class_names,
)

CLASSES = ["mel", "nv", "bkl"]


@pytest.fixture(scope="module")
def tiny_model():
    import keras
    from keras import layers

    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(4, 3, padding="same", name="gradcam_target")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inputs, outputs)


@pytest.fixture()
def sample_image():
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (80, 60, 3), dtype=np.uint8))


# --- Confidence / risk ---------------------------------------------------
@pytest.mark.parametrize("value,band", [
    (0.99, "High"), (0.75, "High"), (0.7499, "Moderate"), (0.50, "Moderate"),
    (0.4999, "Low"), (0.0, "Low"),
])
def test_confidence_bands(value, band):
    assert confidence.confidence_band(value) == band


def test_low_confidence_is_always_uncertain():
    assert confidence.assess_risk("mel", 0.3) == constants.RISK_UNCERTAIN
    assert confidence.assess_risk("nv", 0.3) == constants.RISK_UNCERTAIN


def test_risk_follows_class_when_confident():
    assert confidence.assess_risk("mel", 0.9) == constants.RISK_PRIORITY
    assert confidence.assess_risk("nv", 0.9) == constants.RISK_ROUTINE
    assert confidence.assess_risk("unknown_class", 0.9) == constants.RISK_MONITOR


def test_guidance_always_recommends_professional_for_uncertain():
    text = confidence.next_steps_guidance(constants.RISK_UNCERTAIN, "Low")
    assert "dermatologist" in text.lower()


# --- Preprocessing -------------------------------------------------------
def test_preprocess_shape_and_range(sample_image):
    batch = preprocess_image(sample_image, (32, 32))
    assert batch.shape == (1, 32, 32, 3) and batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_preprocess_converts_non_rgb():
    gray = Image.new("L", (50, 50), 128)
    assert preprocess_image(gray, (16, 16)).shape == (1, 16, 16, 3)


# --- Predictor -----------------------------------------------------------
def test_predict_output_contract(tiny_model, sample_image):
    result = Predictor(tiny_model, CLASSES).predict(sample_image)
    for key in ("condition", "confidence", "risk_level", "probabilities"):
        assert key in result
    probs = result["probabilities"]
    assert len(probs) == len(CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    # Confidence is exactly the model's top probability, not an invented value.
    assert result["confidence"] == pytest.approx(max(probs.values()))
    assert result["top_predictions"][0]["probability"] == pytest.approx(result["confidence"])
    assert result["risk_level"] in {
        constants.RISK_ROUTINE, constants.RISK_MONITOR, constants.RISK_PRIORITY, constants.RISK_UNCERTAIN,
    }


def test_predict_matches_raw_model_output(tiny_model, sample_image):
    predictor = Predictor(tiny_model, CLASSES)
    expected = np.asarray(tiny_model(preprocess_image(sample_image, (32, 32)), training=False))[0]
    result = predictor.predict(sample_image)
    assert result["condition_id"] == CLASSES[int(expected.argmax())]
    assert result["confidence"] == pytest.approx(float(expected.max()), abs=1e-6)


def test_class_count_mismatch_rejected(tiny_model):
    with pytest.raises(ClassNamesError):
        Predictor(tiny_model, ["a", "b"])


def test_gradcam_returns_three_images(tiny_model, sample_image):
    result = Predictor(tiny_model, CLASSES).explain(sample_image)
    assert result is not None
    original, heatmap, overlay = result
    assert original.size == heatmap.size == overlay.size == (32, 32)


def test_missing_model_reports_clearly(tmp_path):
    ok, message = get_model_status(tmp_path / "none.keras", tmp_path / "none.json")
    assert not ok and "train" in message.lower()
    with pytest.raises(ModelNotAvailableError):
        Predictor.load(tmp_path / "none.keras", tmp_path / "none.json")


def test_corrupt_model_file_reports_clearly(tmp_path):
    model_path = tmp_path / "m.keras"
    model_path.write_bytes(b"not a model")
    names = tmp_path / "c.json"
    names.write_text(json.dumps(CLASSES))
    with pytest.raises(ModelNotAvailableError):
        Predictor.load(model_path, names)


@pytest.mark.parametrize("content", ["not json", "{}", '["only_one"]', '["a","a"]', '[1,2]'])
def test_bad_class_names_rejected(tmp_path, content):
    path = tmp_path / "c.json"
    path.write_text(content)
    with pytest.raises(ClassNamesError):
        load_class_names(path)


def test_missing_class_names_rejected(tmp_path):
    with pytest.raises(ClassNamesError):
        load_class_names(tmp_path / "missing.json")


def test_save_load_roundtrip(tiny_model, tmp_path, sample_image):
    model_path, names_path = tmp_path / "m.keras", tmp_path / "c.json"
    tiny_model.save(model_path)
    names_path.write_text(json.dumps(CLASSES))
    loaded = Predictor.load(model_path, names_path)
    assert loaded.predict(sample_image)["condition_id"] in CLASSES


def test_invalid_model_output_is_rejected(sample_image):
    class BadModel:
        output_shape = (None, 3)
        input_shape = (None, 32, 32, 3)

        def __call__(self, batch, training=False):
            return np.array([[5.0, -1.0, 2.0]])  # not a probability distribution

    with pytest.raises(PredictionError):
        Predictor(BadModel(), CLASSES).predict(sample_image)


# --- Training pipeline (tiny synthetic data, temp dirs) ------------------
def _make_synthetic_dataset(root, per_class=12):
    rng = np.random.default_rng(0)
    base_colors = {"mel": (200, 40, 40), "nv": (40, 200, 40), "bkl": (40, 40, 200)}
    for label, color in base_colors.items():
        (root / label).mkdir(parents=True)
        for i in range(per_class):
            noise = rng.integers(-30, 30, (48, 48, 3))
            arr = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / label / f"{i}.png")


def test_prepare_dataset_splits_without_leakage(tmp_path):
    from training.preprocess import prepare_dataset

    raw = tmp_path / "raw"
    _make_synthetic_dataset(raw)
    summary = prepare_dataset(raw, tmp_path / "proc", tmp_path / "meta", image_size=(32, 32))
    assert summary.class_names == sorted(CLASSES)
    import pandas as pd

    manifest = pd.read_csv(summary.manifest_path)
    assert set(manifest.split) == {"train", "val", "test"}
    assert manifest.path.is_unique


def test_empty_and_tiny_datasets_rejected(tmp_path):
    from training.preprocess import DatasetError, EmptyDatasetError, prepare_dataset

    with pytest.raises(EmptyDatasetError):
        prepare_dataset(tmp_path / "missing", tmp_path / "p", tmp_path / "m")
    (tmp_path / "raw" / "a").mkdir(parents=True)
    with pytest.raises(EmptyDatasetError):
        prepare_dataset(tmp_path / "raw", tmp_path / "p", tmp_path / "m")
    Image.new("RGB", (40, 40)).save(tmp_path / "raw" / "a" / "1.png")
    with pytest.raises(DatasetError):
        prepare_dataset(tmp_path / "raw", tmp_path / "p", tmp_path / "m")


def test_compute_metrics_values():
    from training.evaluate import compute_metrics

    metrics = compute_metrics(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 2]), ["a", "b", "c"])
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
    assert metrics["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["a"]["recall"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]), ["a", "b"])


def test_load_metrics_absent_or_invalid_returns_none(tmp_path):
    from training.evaluate import load_metrics

    assert load_metrics(tmp_path / "none.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert load_metrics(bad) is None


def test_end_to_end_training_and_inference(tmp_path, monkeypatch):
    """Preprocess -> train (tiny CNN, 1 epoch) -> evaluate -> load in Predictor -> predict."""
    from training import train as train_module
    from training.preprocess import prepare_dataset

    raw = tmp_path / "raw"
    _make_synthetic_dataset(raw)
    models_dir = tmp_path / "models"
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "proc")
    monkeypatch.setattr(config, "METADATA_DIR", tmp_path / "meta")
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "MODEL_PATH", models_dir / "skin_model.keras")
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", models_dir / "class_names.json")
    monkeypatch.setattr(config, "METRICS_PATH", models_dir / "model_metrics.json")
    monkeypatch.setattr(config, "HISTORY_PATH", models_dir / "training_history.json")

    prepare_dataset(raw, image_size=(32, 32))
    metrics = train_module.train("cnn", epochs=2, fine_tune_epochs=0, batch_size=8,
                                 learning_rate=1e-3, pretrained=False, image_size=(32, 32))
    saved = json.loads((models_dir / "model_metrics.json").read_text())
    assert saved["accuracy"] == metrics["accuracy"]
    assert saved["num_test_samples"] == sum(
        sum(row) for row in saved["confusion_matrix"])
    assert saved["model"]["architecture"] == "cnn"

    predictor = Predictor.load()
    image = Image.fromarray(np.full((40, 40, 3), (200, 40, 40), dtype=np.uint8))
    result = predictor.predict(image)
    assert result["condition_id"] in predictor.class_names
    assert predictor.explain(image) is not None
