from __future__ import annotations

from spittrack.data.contracts import ActionInterval
from spittrack.data.tracking import TrackFrame
from spittrack.common.errors import WindowError
from spittrack.data.folds import FoldManifest
from spittrack.data.windows import build_window_index, select_partition_windows


def interval() -> ActionInterval:
    return ActionInterval(
        source_id="S01",
        tracking_run_id="run42",
        track_id=7,
        actor_role="main_actor",
        action_label="spitting",
        start_s=1.0,
        end_s=2.0,
        negative_subtype=None,
        annotation_confidence=1.0,
        notes="",
    )


def frame(index: int) -> TrackFrame:
    return TrackFrame(
        source_id="S01",
        tracking_run_id="run42",
        frame_index=index,
        timestamp_s=index / 25,
        track_id=7,
        bbox_xyxy=(10, 10, 100, 200),
        det_conf=0.9,
    )


def test_complete_track_emits_expected_half_second_stride(source_factory) -> None:
    source = source_factory("S01", True, duration_s=3.0)
    windows, summary = build_window_index([source], [frame(i) for i in range(75)], [interval()])
    assert [item.start_frame for item in windows] == [0, 13, 25]
    assert all(len(item.sampled_frame_indices) == 16 for item in windows)
    assert summary.emitted_windows == 3
    assert summary.rejected_incomplete_windows == 0


def test_missing_track_frame_rejects_window_without_interpolation(source_factory) -> None:
    source = source_factory("S01", True, duration_s=2.0)
    frames = [frame(i) for i in range(50) if i != 20]
    windows, summary = build_window_index([source], frames, [interval()])
    assert windows == []
    assert summary.rejected_incomplete_windows == 1


def test_partition_selection_excludes_ignore_and_rejects_unassigned_sources(source_factory) -> None:
    source = source_factory("S01", True, duration_s=3.0)
    windows, _ = build_window_index([source], [frame(i) for i in range(75)], [interval()])
    manifest = FoldManifest(
        fold=1,
        train=("S01",),
        validation=("S02",),
        test=("S03",),
        external_real=("R01",),
        seed=42,
    )
    selected = select_partition_windows(windows, manifest, "train")
    assert all(item.label != "ignore" for item in selected)

    invalid_manifest = FoldManifest(
        fold=1,
        train=("S99",),
        validation=("S02",),
        test=("S03",),
        external_real=("R01",),
        seed=42,
    )
    import pytest

    with pytest.raises(WindowError, match="not assigned"):
        select_partition_windows(windows, invalid_manifest, "train")
