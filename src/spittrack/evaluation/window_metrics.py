"""Binary window metrics and validation-only threshold selection."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from spittrack.data.contracts import NEGATIVE_SUBTYPES


def select_f1_threshold(labels: Iterable[int], probabilities: Iterable[float]) -> tuple[float, float]:
    """Choose the validation threshold with maximum F1; ties prefer the higher threshold."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(probabilities), dtype=np.float64)
    if y_true.size == 0 or scores.size != y_true.size:
        raise ValueError("labels and probabilities must be non-empty and equal length")
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("labels must be binary")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        predicted = (scores >= threshold).astype(np.int64)
        f1 = float(f1_score(y_true, predicted, zero_division=0))
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_threshold = float(threshold)
            best_f1 = f1
    return best_threshold, best_f1


def binary_window_metrics(
    labels: Iterable[int], probabilities: Iterable[float], *, threshold: float
) -> dict[str, object]:
    """Compute only metrics supported by supplied predictions; never fill missing values."""
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_curve,
        precision_score,
        recall_score,
        f1_score,
        auc,
    )

    y_true = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(probabilities), dtype=np.float64)
    if y_true.size == 0 or scores.size != y_true.size:
        raise ValueError("labels and probabilities must be non-empty and equal length")
    predicted = (scores >= threshold).astype(np.int64)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    return {
        "threshold": float(threshold),
        "count": int(y_true.size),
        "positive_count": int(y_true.sum()),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def hard_negative_false_positive_counts(
    labels: Iterable[int],
    probabilities: Iterable[float],
    negative_subtypes: Iterable[str],
    *,
    threshold: float,
) -> dict[str, int]:
    """Count thresholded false-positive windows by the declared hard-negative subtype."""
    counts = {subtype: 0 for subtype in sorted(NEGATIVE_SUBTYPES)}
    triples = list(zip(labels, probabilities, negative_subtypes))
    for label, probability, subtype in triples:
        if int(label) == 0 and float(probability) >= threshold:
            key = subtype or "normal_activity"
            if key not in counts:
                raise ValueError(f"Unknown hard-negative subtype in predictions: {key!r}")
            counts[key] += 1
    return counts
