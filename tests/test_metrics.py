from __future__ import annotations

import pytest

from spittrack.evaluation.window_metrics import binary_window_metrics, select_f1_threshold


def test_threshold_and_metrics_on_hand_computable_arrays() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.6, 0.9]
    threshold, f1 = select_f1_threshold(labels, probabilities)
    assert threshold == pytest.approx(0.6)
    assert f1 == pytest.approx(1.0)
    metrics = binary_window_metrics(labels, probabilities, threshold=threshold)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]

