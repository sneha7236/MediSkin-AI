"""Model analytics: every number comes from models/model_metrics.json written by training."""
from __future__ import annotations

import streamlit as st

from config import config
from modules import analytics
from utils import ui
from utils.helpers import percent


def _model_details(metrics: dict) -> None:
    model = metrics.get("model") or {}
    dataset = metrics.get("dataset") or {}
    parts = []
    if model.get("architecture"):
        pretrained = "ImageNet-pretrained" if model.get("pretrained_weights") else "random initialisation"
        parts.append(f"Architecture: {model['architecture']} ({pretrained})")
    if dataset:
        parts.append("Images: " + ", ".join(f"{k} {v}" for k, v in dataset.items()))
    parts.append(f"Evaluated on {metrics.get('num_test_samples', '?')} held-out test images")
    if metrics.get("generated_at"):
        parts.append(f"Generated {metrics['generated_at']}")
    st.caption(" · ".join(parts))


def render() -> None:
    ui.page_header("Model Analytics", "How the classifier performed on held-out test images.", "monitoring")

    metrics = analytics.load_metrics()
    if metrics is None:
        ui.empty_state(
            "No evaluation metrics yet",
            "Metrics are produced by training. Run `python -m training.train` (see README) and the results "
            f"will appear here. Nothing is estimated or pre-filled. Expected file: {config.METRICS_PATH.name}.",
            "monitoring",
        )
        ui.disclaimer()
        return

    _model_details(metrics)
    for column, (label, value) in zip(st.columns(4, gap="small"), analytics.summary_values(metrics).items()):
        column.html(ui.metric_card(label, percent(value)))

    st.markdown("### Class-wise performance")
    table = analytics.per_class_table(metrics)
    score = st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Precision": score, "Recall": score, "F1-score": score,
        "Test samples": st.column_config.NumberColumn(format="%d"),
    })

    st.markdown("### Confusion matrix")
    left, _ = st.columns([1.3, 1])
    with left:
        st.pyplot(analytics.confusion_matrix_figure(metrics), clear_figure=True)
    st.caption("Rows are the actual class, columns the predicted class.")

    curves = analytics.load_training_history()
    if curves is not None:
        st.markdown("### Training curves")
        accuracy, loss = st.columns(2)
        accuracy_cols = [c for c in curves if "accuracy" in c]
        loss_cols = [c for c in curves if "loss" in c]
        if accuracy_cols:
            accuracy.line_chart(curves[accuracy_cols], height=260)
        if loss_cols:
            loss.line_chart(curves[loss_cols], height=260)
    st.warning("Test-set results describe this model on this dataset only. They do not guarantee accuracy "
               "on your own photos or replace a clinical examination.", icon=":material/info:")
    ui.disclaimer()
