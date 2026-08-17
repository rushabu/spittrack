from __future__ import annotations

import pytest

from spittrack.common.errors import WindowError
from spittrack.data.contracts import ActionInterval
from spittrack.data.windows import label_window, uniform_sample_indices


def spit(start: float, end: float) -> ActionInterval:
    return ActionInterval(
        source_id="S01",
        tracking_run_id="run",
        track_id=1,
        actor_role="main_actor",
        action_label="spitting",
        start_s=start,
        end_s=end,
        negative_subtype=None,
        annotation_confidence=1.0,
        notes="",
    )


def test_uniform_sampling_is_exact_deterministic_and_monotonic() -> None:
    first = uniform_sample_indices(100, 150, 16)
    second = uniform_sample_indices(100, 150, 16)
    assert first == second
    assert len(first) == 16
    assert first[0] == 100
    assert first[-1] == 149
    assert all(left < right for left, right in zip(first, first[1:]))


def test_uniform_sampling_rejects_too_short_interval() -> None:
    with pytest.raises(WindowError, match="Cannot sample"):
        uniform_sample_indices(0, 10, 16)


def test_label_positive_at_exactly_half_action_duration() -> None:
    result = label_window(3.5, 4.5, [spit(4.0, 5.0)])
    assert result.label == "spitting"
    assert result.overlap_fraction_of_action == pytest.approx(0.5)


def test_label_partial_overlap_is_ignore_and_zero_overlap_is_negative() -> None:
    assert label_window(3.9, 4.2, [spit(4.0, 5.0)]).label == "ignore"
    assert label_window(1.0, 3.0, [spit(4.0, 5.0)]).label == "non_spitting"

