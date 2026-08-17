from __future__ import annotations

import pytest

from spittrack.inference.event_aggregator import WindowScore, aggregate_events


def scores(values, *, track=1, starts=None):
    if starts is None:
        starts = [index * 0.5 for index in range(len(values))]
    return [
        WindowScore("S01", track, start, start + 2.0, value, (index, 0, index + 10, 20))
        for index, (start, value) in enumerate(zip(starts, values))
    ]


def test_isolated_spike_does_not_create_event() -> None:
    assert aggregate_events(scores([0.1, 0.1, 0.9, 0.1, 0.1]), threshold=0.5) == []


def test_two_consecutive_smoothed_positives_create_event() -> None:
    events = aggregate_events(scores([0.1, 0.8, 0.9, 0.8, 0.1]), threshold=0.5)
    assert len(events) == 1
    assert events[0].track_id == 1


def test_gap_at_most_one_second_merges_and_larger_gap_splits() -> None:
    first = scores([0.9, 0.9], starts=[0.0, 0.5])
    near = scores([0.9, 0.9], starts=[3.5, 4.0])
    far = scores([0.9, 0.9], starts=[5.1, 5.6])
    assert len(aggregate_events(first + near, threshold=0.5)) == 1
    assert len(aggregate_events(first + far, threshold=0.5)) == 2


def test_different_tracks_never_merge() -> None:
    events = aggregate_events(scores([0.9, 0.9], track=1) + scores([0.9, 0.9], track=2), threshold=0.5)
    assert len(events) == 2
    assert {event.track_id for event in events} == {1, 2}


def test_confidence_is_mean_of_top_three_smoothed_scores() -> None:
    events = aggregate_events(scores([0.6, 0.7, 0.8, 0.9]), threshold=0.5)
    assert len(events) == 1
    assert events[0].confidence == pytest.approx((0.9 + 0.8 + 0.7) / 3)

