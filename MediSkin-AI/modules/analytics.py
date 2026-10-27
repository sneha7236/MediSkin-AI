"""Model-analytics helpers. Everything shown comes from models/model_metrics.json,
which is written by ``training.evaluate``; nothing is hardcoded or estimated here."""
from __future__ import annotations

import json

import pandas as pd
from matplotlib.figure import Figure

from config import config
from modules.disease_info import display_name
from training.evaluate import load_metrics

__all__ = ["load_metrics", "per_class_table", "confusion_matrix_figure", "summary_values",
           "load_training_history"]

_CURVES = {"accuracy": "Training accuracy", "val_accuracy": "Validation accuracy",
           "loss": "Training loss", "val_loss": "Validation loss"}


def load_training_history() -> pd.DataFrame | None:
    """Per-epoch curves saved by training (stages concatenated), or None if unavailable."""
    path = config.HISTORY_PATH
    if not path.is_file():
        return None
    try:
        stages = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stages, dict):
            return None
        frames = [pd.DataFrame({_CURVES[k]: v for k, v in stage.items() if k in _CURVES})
                  for stage in stages.values() if isinstance(stage, dict)]
        frame = pd.concat(frames, ignore_index=True) if frames else None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    if frame is None or frame.empty:
        return None
    frame.index = range(1, len(frame) + 1)
    frame.index.name = "Epoch"
    return frame


def summary_values(metrics: dict) -> dict[str, float]:
    """Headline numbers: accuracy plus macro precision/recall/F1."""
    return {
        "Accuracy": metrics["accuracy"],
        "Precision (macro)": metrics["macro"]["precision"],
        "Recall (macro)": metrics["macro"]["recall"],
        "F1-score (macro)": metrics["macro"]["f1"],
    }


def per_class_table(metrics: dict) -> pd.DataFrame:
    rows = [
        {"Class": display_name(name), "Precision": v["precision"], "Recall": v["recall"],
         "F1-score": v["f1"], "Test samples": v["support"]}
        for name, v in metrics["per_class"].items()
    ]
    return pd.DataFrame(rows)


def confusion_matrix_figure(metrics: dict) -> Figure:
    labels = [display_name(n) for n in metrics["class_names"]]
    matrix = metrics["confusion_matrix"]
    n = len(labels)
    fig = Figure(figsize=(max(4.5, 0.9 * n + 2), max(4, 0.8 * n + 1.5)), dpi=110)
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n), labels=labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels=labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    peak = max(max(row) for row in matrix) or 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=8,
                    color="white" if matrix[i][j] > peak / 2 else "#0F172A")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
