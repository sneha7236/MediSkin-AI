"""Model evaluation on the held-out test split. Writes real metrics to models/model_metrics.json.

Usage:
    python -m training.evaluate
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import config
from utils.logger import get_logger

logger = get_logger("evaluate")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy, macro/weighted precision-recall-F1, per-class scores, confusion matrix."""
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate on an empty test set.")
    labels = list(range(len(class_names)))
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_class": {
            name: {"precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f[i]), "support": int(per_s[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "num_test_samples": int(len(y_true)),
        "class_names": list(class_names),
    }


def predict_dataset(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over a (images, int_labels) tf.data dataset; return (y_true, y_pred)."""
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model(images, training=False).numpy()
        y_pred.append(probs.argmax(axis=1))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def save_metrics(metrics: dict, extra: dict | None = None, path: Path | None = None) -> Path:
    path = Path(path or config.METRICS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {}), **metrics}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Metrics saved (accuracy=%.4f, n=%d)", metrics["accuracy"], metrics["num_test_samples"])
    return path


def load_metrics(path: Path | None = None) -> dict | None:
    """Metrics saved by training, or None if absent/unreadable. Never fabricates values."""
    path = Path(path or config.METRICS_PATH)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"accuracy", "macro", "per_class", "confusion_matrix", "class_names"}
    return data if required <= data.keys() else None


def main() -> None:
    """Re-evaluate the saved model on the test split of the prepared dataset."""
    import argparse

    from training.preprocess import DatasetError

    argparse.ArgumentParser(description="Re-evaluate models/skin_model.keras on the prepared test split "
                                        "and rewrite models/model_metrics.json.").parse_args()
    try:
        _evaluate_saved_model()
    except DatasetError as exc:
        raise SystemExit(f"ERROR: {exc}")


def _evaluate_saved_model() -> None:
    import keras

    from training.preprocess import DatasetError, load_manifest
    from training.train import make_dataset, read_class_names

    if not config.MODEL_PATH.is_file():
        raise DatasetError(f"Model file not found: {config.MODEL_PATH}. Train a model first.")
    class_names = read_class_names()
    manifest = load_manifest()
    test_df = manifest[manifest.split == "test"]
    if test_df.empty:
        raise DatasetError("The prepared dataset has no test images. Run `python -m training.preprocess`.")
    model = keras.models.load_model(config.MODEL_PATH, compile=False)
    size = (model.input_shape[2], model.input_shape[1])
    dataset = make_dataset(test_df, class_names, batch_size=32, image_size=size, training=False)
    y_true, y_pred = predict_dataset(model, dataset)
    metrics = compute_metrics(y_true, y_pred, class_names)
    path = save_metrics(metrics, extra=_previous_extra())
    print(f"Accuracy: {metrics['accuracy']:.4f}  (n={metrics['num_test_samples']})  -> {path}")


def _previous_extra() -> dict:
    previous = load_metrics()
    return {k: previous[k] for k in ("model", "dataset") if previous and k in previous}


if __name__ == "__main__":
    main()
