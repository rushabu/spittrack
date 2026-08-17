from __future__ import annotations

import pytest

from spittrack.evaluation.event_metrics import GroundTruthEvent, event_metrics, temporal_iou
from spittrack.inference.event_aggregator import SpittingEvent


def predicted(source: str, track: int, start: float, end: float) -> SpittingEvent:
    return SpittingEvent(source, track, start, end, 0.9, (0, 0, 10, 20), 2)


def test_temporal_iou_is_hand_computable() -> None:
    assert temporal_iou(0, 2, 1, 3) == pytest.approx(1 / 3)
    assert temporal_iou(0, 2, 0.5, 2.5) == pytest.approx(1.5 / 2.5)


def test_event_metrics_match_once_and_score_wrong_track_separately() -> None:
    predictions = [predicted("S01", 9, 1.0, 3.0), predicted("S02", 1, 0, 1)]
    truth = [GroundTruthEvent("S01", 4, 1.1, 2.9)]
    metrics = event_metrics(
        predictions,
        truth,
        source_durations_s={"S01": 10, "S02": 30},
        no_spit_source_ids={"S02"},
        attribution_source_ids={"S01"},
    )
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["false_positives_per_minute"] == pytest.approx(2.0)
    assert metrics["track_attribution_accuracy"] == pytest.approx(0.0)


def test_no_matches_report_timing_error_as_missing_not_fabricated_zero() -> None:
    metrics = event_metrics(
        [],
        [GroundTruthEvent("S01", 1, 1, 2)],
        source_durations_s={"S01": 10},
        no_spit_source_ids=set(),
    )
    assert metrics["mean_start_error_s"] is None
    assert metrics["mean_end_error_s"] is None
