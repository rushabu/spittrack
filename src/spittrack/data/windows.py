"""Pure temporal sampling, labeling, and track-window indexing."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

from spittrack.common.errors import WindowError
from spittrack.data.contracts import ActionInterval, SourceRecord
from spittrack.data.tracking import TrackFrame
from spittrack.data.folds import FoldManifest

LOGGER = logging.getLogger(__name__)
WINDOW_LABELS = frozenset({"spitting", "non_spitting", "ignore"})


@dataclass(frozen=True)
class WindowLabel:
    """Label and maximum annotated-action overlap for one temporal window."""

    label: str
    overlap_fraction_of_action: float


@dataclass(frozen=True)
class WindowRecord:
    """A deterministic, complete per-track temporal window."""

    source_id: str
    tracking_run_id: str
    track_id: int
    start_frame: int
    end_frame_exclusive: int
    start_s: float
    end_s: float
    sampled_frame_indices: tuple[int, ...]
    label: str
    overlap_fraction_of_action: float
    negative_subtype: str | None

    @property
    def window_id(self) -> str:
        return f"{self.source_id}_track{self.track_id:04d}_start{self.start_frame:08d}"


@dataclass(frozen=True)
class WindowBuildSummary:
    """Counts that preserve the distinction between sources, tracks, and windows."""

    source_count: int
    track_count: int
    emitted_windows: int
    ignored_windows: int
    rejected_incomplete_windows: int


def uniform_sample_indices(start: int, end_exclusive: int, count: int = 16) -> tuple[int, ...]:
    """Uniformly sample deterministic, strictly increasing indices including both endpoints."""
    if start < 0 or end_exclusive <= start:
        raise WindowError(f"Invalid frame interval [{start}, {end_exclusive})")
    length = end_exclusive - start
    if count < 2:
        raise WindowError(f"sample count must be at least 2, got {count}")
    if length < count:
        raise WindowError(f"Cannot sample {count} unique frames from a {length}-frame interval")
    last_offset = length - 1
    denominator = count - 1
    indices = tuple(start + (index * last_offset + denominator // 2) // denominator for index in range(count))
    if len(indices) != count or any(left >= right for left, right in zip(indices, indices[1:])):
        raise WindowError(f"Uniform sampler failed strict monotonicity: {indices}")
    return indices


def label_window(
    start_s: float,
    end_s: float,
    spitting_intervals: Iterable[ActionInterval],
    *,
    positive_fraction: float = 0.5,
) -> WindowLabel:
    """Apply the frozen action-duration overlap rule to one window."""
    if start_s < 0 or end_s <= start_s:
        raise WindowError(f"Invalid window interval [{start_s}, {end_s})")
    if not 0.0 < positive_fraction <= 1.0:
        raise WindowError("positive_fraction must be in (0, 1]")
    maximum = 0.0
    for interval in spitting_intervals:
        if interval.action_label != "spitting":
            continue
        overlap = max(0.0, min(end_s, interval.end_s) - max(start_s, interval.start_s))
        maximum = max(maximum, overlap / interval.duration_s)
    if maximum >= positive_fraction:
        return WindowLabel("spitting", maximum)
    if maximum == 0.0:
        return WindowLabel("non_spitting", 0.0)
    return WindowLabel("ignore", maximum)


def _stride_start_frames(first: int, last_start: int, stride_frames: Decimal) -> Iterable[int]:
    step = 0
    previous: int | None = None
    while True:
        offset = int((Decimal(step) * stride_frames).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        current = first + offset
        if current > last_start:
            return
        if current != previous:
            yield current
            previous = current
        step += 1


def build_window_index(
    sources: Iterable[SourceRecord],
    track_frames: Iterable[TrackFrame],
    annotations: Iterable[ActionInterval],
    *,
    fps: int = 25,
    window_seconds: float = 2.0,
    stride_seconds: float = 0.5,
    sampled_frames: int = 16,
    positive_fraction: float = 0.5,
) -> tuple[list[WindowRecord], WindowBuildSummary]:
    """Build windows only from complete, non-interpolated per-track frame sequences."""
    if fps <= 0 or window_seconds <= 0 or stride_seconds <= 0:
        raise WindowError("FPS, window duration, and stride must be positive")
    window_frame_decimal = Decimal(str(window_seconds)) * Decimal(fps)
    if window_frame_decimal != window_frame_decimal.to_integral_value():
        raise WindowError("window_seconds * fps must be an integer number of frames")
    window_frames = int(window_frame_decimal)
    stride_frames = Decimal(str(stride_seconds)) * Decimal(fps)
    if stride_frames < 1:
        raise WindowError("stride must advance by at least one frame")

    source_map = {source.source_id: source for source in sources}
    grouped_frames: dict[tuple[str, int], list[TrackFrame]] = defaultdict(list)
    for frame in track_frames:
        if frame.source_id not in source_map:
            raise WindowError(f"Tracking frame references unknown source {frame.source_id!r}")
        grouped_frames[(frame.source_id, frame.track_id)].append(frame)
    intervals_by_track: dict[tuple[str, int], list[ActionInterval]] = defaultdict(list)
    negative_intervals_by_track: dict[tuple[str, int], list[ActionInterval]] = defaultdict(list)
    for interval in annotations:
        key = (interval.source_id, interval.track_id)
        if interval.action_label == "spitting":
            intervals_by_track[key].append(interval)
        else:
            negative_intervals_by_track[key].append(interval)

    records: list[WindowRecord] = []
    rejected_incomplete = 0
    ignored = 0
    for key in sorted(grouped_frames):
        frames = sorted(grouped_frames[key], key=lambda item: item.frame_index)
        by_index = {frame.frame_index: frame for frame in frames}
        first = frames[0].frame_index
        last_start = frames[-1].frame_index - window_frames + 1
        if last_start < first:
            LOGGER.info("Rejecting track %s: duration is shorter than %.2fs", key, window_seconds)
            continue
        for start_frame in _stride_start_frames(first, last_start, stride_frames):
            end_frame = start_frame + window_frames
            required = range(start_frame, end_frame)
            if any(index not in by_index for index in required):
                rejected_incomplete += 1
                LOGGER.warning(
                    "Rejecting incomplete window source=%s track=%s start_frame=%s; no interpolation",
                    key[0],
                    key[1],
                    start_frame,
                )
                continue
            start_s = start_frame / fps
            end_s = end_frame / fps
            label = label_window(
                start_s,
                end_s,
                intervals_by_track[key],
                positive_fraction=positive_fraction,
            )
            if label.label == "ignore":
                ignored += 1
            overlapping_negatives = [
                interval
                for interval in negative_intervals_by_track[key]
                if min(end_s, interval.end_s) > max(start_s, interval.start_s)
            ]
            negative_subtype = (
                overlapping_negatives[0].negative_subtype or overlapping_negatives[0].action_label
                if overlapping_negatives
                else source_map[key[0]].negative_subtype
            )
            run_ids = {by_index[index].tracking_run_id for index in required}
            if len(run_ids) != 1:
                raise WindowError(f"Window {key} at frame {start_frame} mixes tracking_run_id values")
            records.append(
                WindowRecord(
                    source_id=key[0],
                    tracking_run_id=run_ids.pop(),
                    track_id=key[1],
                    start_frame=start_frame,
                    end_frame_exclusive=end_frame,
                    start_s=start_s,
                    end_s=end_s,
                    sampled_frame_indices=uniform_sample_indices(start_frame, end_frame, sampled_frames),
                    label=label.label,
                    overlap_fraction_of_action=label.overlap_fraction_of_action,
                    negative_subtype=negative_subtype,
                )
            )
    summary = WindowBuildSummary(
        source_count=len({record.source_id for record in records}),
        track_count=len(grouped_frames),
        emitted_windows=len(records),
        ignored_windows=ignored,
        rejected_incomplete_windows=rejected_incomplete,
    )
    return records, summary


def write_window_index(records: Iterable[WindowRecord], output_path: Path) -> None:
    """Write a fold-independent window index; fold roles are joined from immutable manifests at runtime."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_id",
        "source_id",
        "tracking_run_id",
        "track_id",
        "start_frame",
        "end_frame_exclusive",
        "start_s",
        "end_s",
        "sampled_frame_indices",
        "label",
        "overlap_fraction_of_action",
        "negative_subtype",
        "clip_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "window_id": record.window_id,
                    "source_id": record.source_id,
                    "tracking_run_id": record.tracking_run_id,
                    "track_id": record.track_id,
                    "start_frame": record.start_frame,
                    "end_frame_exclusive": record.end_frame_exclusive,
                    "start_s": f"{record.start_s:.6f}",
                    "end_s": f"{record.end_s:.6f}",
                    "sampled_frame_indices": json.dumps(record.sampled_frame_indices),
                    "label": record.label,
                    "overlap_fraction_of_action": f"{record.overlap_fraction_of_action:.6f}",
                    "negative_subtype": record.negative_subtype or "",
                    "clip_path": f"clips/{record.source_id}/{record.window_id}.npy",
                }
            )


def select_partition_windows(
    records: Iterable[WindowRecord],
    manifest: FoldManifest,
    partition: str,
    *,
    include_ignore: bool = False,
) -> list[WindowRecord]:
    """Join windows to an immutable source manifest without permitting unassigned sources."""
    partition_sources = {
        "train": set(manifest.train),
        "validation": set(manifest.validation),
        "test": set(manifest.test),
    }
    if partition not in partition_sources:
        raise WindowError(f"partition must be one of {sorted(partition_sources)}, got {partition!r}")
    manifest.validate()
    assigned = set().union(*partition_sources.values())
    result: list[WindowRecord] = []
    for record in records:
        if record.source_id in manifest.external_real:
            raise WindowError(
                f"External real source {record.source_id} appeared in a synthetic window collection"
            )
        if record.source_id not in assigned:
            raise WindowError(
                f"Window source {record.source_id} is not assigned by fold {manifest.fold}; refusing leakage"
            )
        if record.source_id in partition_sources[partition]:
            if include_ignore or record.label != "ignore":
                result.append(record)
    return result
