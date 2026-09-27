"""Classification metrics and prediction helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    num_classes: int,
    confidences: Iterable[float] | None = None,
) -> dict:
    y_true = np.asarray(list(targets), dtype=np.int64)
    y_pred = np.asarray(list(predictions), dtype=np.int64)
    labels = np.arange(num_classes)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_precision": per_precision.tolist(),
        "per_class_recall": per_recall.tolist(),
        "per_class_f1": per_f1.tolist(),
        "support": support.astype(int).tolist(),
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    if confidences is not None:
        confidence_values = np.asarray(list(confidences), dtype=np.float64)
        if len(confidence_values) != len(y_true):
            raise ValueError("Confidence and target lengths differ")
        correct = (y_true == y_pred).astype(np.float64)
        edges = np.linspace(0.0, 1.0, 16)
        ece = 0.0
        for lower, upper in zip(edges[:-1], edges[1:]):
            in_bin = (confidence_values > lower) & (confidence_values <= upper)
            if in_bin.any():
                ece += float(in_bin.mean()) * abs(
                    float(correct[in_bin].mean()) - float(confidence_values[in_bin].mean())
                )
        result["expected_calibration_error_15bin"] = ece
        result["mean_confidence"] = float(confidence_values.mean())
    return result
