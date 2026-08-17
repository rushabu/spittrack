"""One-to-one temporal event matching, timing error, FP/min, and attribution metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from spittrack.inference.event_aggregator import SpittingEvent


@dataclass(frozen=True)
class GroundTruthEvent:
    source_id: str
    track_id: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class EventMatch:
    predicted_index: int
    ground_truth_index: int
    temporal_iou: float


def temporal_iou(
    first_start: float, first_end: float, second_start: float, second_end: float
) -> float:
    if first_end <= first_start or second_end <= second_start:
        raise ValueError("Event intervals must have positive duration")
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union


def match_events(
    predictions: Iterable[SpittingEvent],
    ground_truth: Iterable[GroundTruthEvent],
    *,
    minimum_iou: float = 0.5,
) -> list[EventMatch]:
    """Greedily take highest-IoU source-local pairs; track correctness is scored separately."""
    predicted = list(predictions)
    truth = list(ground_truth)
    candidates: list[tuple[float, int, int]] = []
    for predicted_index, event in enumerate(predicted):
        for truth_index, target in enumerate(truth):
            if event.source_id != target.source_id:
                continue
            overlap = temporal_iou(event.start_s, event.end_s, target.start_s, target.end_s)
            if overlap >= minimum_iou:
                candidates.append((overlap, predicted_index, truth_index))
    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    matches: list[EventMatch] = []
    for overlap, predicted_index, truth_index in sorted(candidates, reverse=True):
        if predicted_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions.add(predicted_index)
        matched_truth.add(truth_index)
        matches.append(EventMatch(predicted_index, truth_index, overlap))
    return matches


def event_metrics(
    predictions: Iterable[SpittingEvent],
    ground_truth: Iterable[GroundTruthEvent],
    *,
    source_durations_s: Mapping[str, float],
    no_spit_source_ids: set[str],
    attribution_source_ids: set[str] | None = None,
    minimum_iou: float = 0.5,
) -> dict[str, float | int | None]:
    predicted = list(predictions)
    truth = list(ground_truth)
    matches = match_events(predicted, truth, minimum_iou=minimum_iou)
    true_positives = len(matches)
    false_positives = len(predicted) - true_positives
    false_negatives = len(truth) - true_positives
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    start_errors = [
        abs(predicted[item.predicted_index].start_s - truth[item.ground_truth_index].start_s)
        for item in matches
    ]
    end_errors = [
        abs(predicted[item.predicted_index].end_s - truth[item.ground_truth_index].end_s)
        for item in matches
    ]
    no_spit_predictions = sum(
        1 for event in predicted if event.source_id in no_spit_source_ids
    )
    no_spit_minutes = sum(
        source_durations_s[source_id]
        for source_id in no_spit_source_ids
        if source_id in source_durations_s
    ) / 60.0
    if no_spit_source_ids and no_spit_minutes <= 0:
        raise ValueError("No-spit source duration must be positive for FP/min")
    attribution_candidates = [
        item
        for item in matches
        if attribution_source_ids is not None
        and truth[item.ground_truth_index].source_id in attribution_source_ids
    ]
    attribution_correct = sum(
        predicted[item.predicted_index].track_id == truth[item.ground_truth_index].track_id
        for item in attribution_candidates
    )
    return {
        "temporal_iou_threshold": minimum_iou,
        "ground_truth_events": len(truth),
        "predicted_events": len(predicted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_start_error_s": sum(start_errors) / len(start_errors) if start_errors else None,
        "mean_end_error_s": sum(end_errors) / len(end_errors) if end_errors else None,
        "false_positives_per_minute": (
            no_spit_predictions / no_spit_minutes if no_spit_minutes > 0 else 0.0
        ),
        "track_attribution_count": len(attribution_candidates),
        "track_attribution_accuracy": (
            attribution_correct / len(attribution_candidates)
            if attribution_candidates
            else None
        ),
    }
