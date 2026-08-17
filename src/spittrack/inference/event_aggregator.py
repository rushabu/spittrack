"""Pure validation-thresholded temporal event aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class WindowScore:
    source_id: str
    track_id: int
    start_s: float
    end_s: float
    probability: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class SmoothedWindowScore:
    window: WindowScore
    smoothed_probability: float


@dataclass(frozen=True)
class SpittingEvent:
    source_id: str
    track_id: int
    start_s: float
    end_s: float
    confidence: float
    peak_bbox_xyxy: tuple[float, float, float, float]
    contributing_window_count: int


def median_smooth_three(
    windows: Iterable[WindowScore], *, maximum_adjacent_start_gap_s: float = 0.5
) -> list[SmoothedWindowScore]:
    """Centered three-window median with endpoint replication, independently per track."""
    if maximum_adjacent_start_gap_s <= 0:
        raise ValueError("maximum_adjacent_start_gap_s must be positive")
    ordered = sorted(windows, key=lambda item: (item.source_id, item.track_id, item.start_s))
    groups: dict[tuple[str, int], list[WindowScore]] = {}
    for window in ordered:
        if not 0.0 <= window.probability <= 1.0:
            raise ValueError(f"Probability must be in [0,1], got {window.probability}")
        if window.end_s <= window.start_s:
            raise ValueError("Window end must be after start")
        groups.setdefault((window.source_id, window.track_id), []).append(window)
    result: list[SmoothedWindowScore] = []
    for key in sorted(groups):
        segments: list[list[WindowScore]] = []
        for window in groups[key]:
            if (
                not segments
                or window.start_s - segments[-1][-1].start_s
                > maximum_adjacent_start_gap_s + 1e-6
            ):
                segments.append([window])
            else:
                segments[-1].append(window)
        for segment in segments:
            for index, window in enumerate(segment):
                left = segment[max(0, index - 1)].probability
                center = window.probability
                right = segment[min(len(segment) - 1, index + 1)].probability
                result.append(SmoothedWindowScore(window, float(median((left, center, right)))))
    return result


def aggregate_events(
    windows: Iterable[WindowScore],
    *,
    threshold: float,
    min_consecutive: int = 2,
    merge_gap_s: float = 1.0,
    expected_stride_s: float = 0.5,
) -> list[SpittingEvent]:
    """Smooth, threshold, require consecutive windows, and merge only within one track."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if min_consecutive < 1 or merge_gap_s < 0 or expected_stride_s <= 0:
        raise ValueError("Invalid event aggregation parameters")
    smoothed = median_smooth_three(
        windows, maximum_adjacent_start_gap_s=expected_stride_s
    )
    groups: dict[tuple[str, int], list[SmoothedWindowScore]] = {}
    for item in smoothed:
        groups.setdefault((item.window.source_id, item.window.track_id), []).append(item)

    events: list[SpittingEvent] = []
    for (source_id, track_id), group in sorted(groups.items()):
        runs: list[list[SmoothedWindowScore]] = []
        current: list[SmoothedWindowScore] = []
        for item in group:
            is_positive = item.smoothed_probability >= threshold
            is_next_stride = (
                not current
                or item.window.start_s - current[-1].window.start_s
                <= expected_stride_s + 1e-6
            )
            if is_positive and is_next_stride:
                current.append(item)
            else:
                if len(current) >= min_consecutive:
                    runs.append(current)
                current = [item] if is_positive else []
        if len(current) >= min_consecutive:
            runs.append(current)

        merged: list[list[SmoothedWindowScore]] = []
        for run in runs:
            if merged:
                prior_end = max(item.window.end_s for item in merged[-1])
                next_start = min(item.window.start_s for item in run)
                if next_start - prior_end <= merge_gap_s + 1e-9:
                    merged[-1].extend(run)
                    continue
            merged.append(list(run))
        for candidate in merged:
            ranked = sorted(
                (item.smoothed_probability for item in candidate), reverse=True
            )
            confidence = sum(ranked[:3]) / min(3, len(ranked))
            peak = max(candidate, key=lambda item: item.smoothed_probability)
            events.append(
                SpittingEvent(
                    source_id=source_id,
                    track_id=track_id,
                    start_s=min(item.window.start_s for item in candidate),
                    end_s=max(item.window.end_s for item in candidate),
                    confidence=confidence,
                    peak_bbox_xyxy=peak.window.bbox_xyxy,
                    contributing_window_count=len(candidate),
                )
            )
    return events
