"""Tracking-manifest contract used by deterministic window construction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from spittrack.common.errors import ContractError
from spittrack.data.contracts import SourceRecord

TRACKING_REQUIRED_COLUMNS = (
    "source_id",
    "tracking_run_id",
    "frame_index",
    "timestamp_s",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "det_conf",
)


@dataclass(frozen=True)
class TrackFrame:
    """One detected trajectory box on one normalized source frame."""

    source_id: str
    tracking_run_id: str
    frame_index: int
    timestamp_s: float
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    det_conf: float


def load_tracking_manifests(
    manifest_dir: Path, sources: Iterable[SourceRecord]
) -> list[TrackFrame]:
    """Load every per-source CSV manifest and enforce source/run/frame integrity."""
    source_map = {source.source_id: source for source in sources}
    if not manifest_dir.is_dir():
        raise ContractError(f"Tracking manifest directory does not exist: {manifest_dir}")
    csv_paths = sorted(manifest_dir.glob("*.csv"))
    if not csv_paths:
        raise ContractError(f"No tracking manifest CSV files found in: {manifest_dir}")

    frames: list[TrackFrame] = []
    observed_sources: set[str] = set()
    run_ids_by_source: dict[str, set[str]] = {}
    seen_keys: set[tuple[str, int, int]] = set()
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ContractError(f"Tracking manifest has no header: {path}")
            missing = sorted(set(TRACKING_REQUIRED_COLUMNS) - set(reader.fieldnames))
            if missing:
                raise ContractError(f"Tracking manifest {path} is missing columns {missing}")
            for row_number, raw in enumerate(reader, start=2):
                row = {key: (value or "").strip() for key, value in raw.items()}
                context = f"{path.name} row {row_number}"
                source_id = row["source_id"]
                if source_id not in source_map:
                    raise ContractError(f"{context}: unknown source_id {source_id!r}")
                run_id = row["tracking_run_id"]
                if not run_id:
                    raise ContractError(f"{context}: tracking_run_id cannot be blank")
                try:
                    frame_index = int(row["frame_index"])
                    timestamp_s = float(row["timestamp_s"])
                    track_id = int(row["track_id"])
                    x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
                    det_conf = float(row["det_conf"])
                except ValueError as exc:
                    raise ContractError(f"{context}: numeric tracking field is invalid") from exc
                if frame_index < 0 or timestamp_s < 0 or track_id < 0:
                    raise ContractError(f"{context}: frame, timestamp, and track ID must be non-negative")
                if x2 <= x1 or y2 <= y1:
                    raise ContractError(f"{context}: bounding box must have positive width and height")
                if not 0.0 <= det_conf <= 1.0:
                    raise ContractError(f"{context}: det_conf must be in [0, 1]")
                expected_timestamp = frame_index / source_map[source_id].fps_normalized
                tolerance = 0.51 / source_map[source_id].fps_normalized
                if abs(timestamp_s - expected_timestamp) > tolerance:
                    raise ContractError(
                        f"{context}: timestamp {timestamp_s} disagrees with frame_index {frame_index} "
                        f"at {source_map[source_id].fps_normalized} FPS"
                    )
                key = (source_id, track_id, frame_index)
                if key in seen_keys:
                    raise ContractError(f"{context}: duplicate source/track/frame key {key}")
                seen_keys.add(key)
                observed_sources.add(source_id)
                run_ids_by_source.setdefault(source_id, set()).add(run_id)
                frames.append(
                    TrackFrame(
                        source_id=source_id,
                        tracking_run_id=run_id,
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        track_id=track_id,
                        bbox_xyxy=(x1, y1, x2, y2),
                        det_conf=det_conf,
                    )
                )
    multiple_runs = {source: runs for source, runs in run_ids_by_source.items() if len(runs) != 1}
    if multiple_runs:
        raise ContractError(f"A source cannot mix tracking_run_id values: {multiple_runs}")
    missing_sources = sorted(set(source_map) - observed_sources)
    if missing_sources:
        raise ContractError(f"Tracking manifests contain no rows for sources: {missing_sources}")
    return frames

