"""CPU-only rendering of tracking manifests over normalized source videos."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from spittrack.common.errors import ContractError
from spittrack.data.tracking import TRACKING_REQUIRED_COLUMNS


@dataclass(frozen=True)
class PreviewBox:
    frame_index: int
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]


def _load_preview_boxes(manifest_path: Path, expected_source_id: str) -> list[PreviewBox]:
    if not manifest_path.is_file():
        raise ContractError(f"Tracking manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ContractError(f"Tracking manifest has no header: {manifest_path}")
        missing = sorted(set(TRACKING_REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ContractError(
                f"Tracking manifest {manifest_path} is missing columns {missing}"
            )
        boxes: list[PreviewBox] = []
        seen: set[tuple[int, int]] = set()
        for row_number, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            context = f"{manifest_path.name} row {row_number}"
            if row["source_id"] != expected_source_id:
                raise ContractError(
                    f"{context}: source_id {row['source_id']!r} does not match manifest "
                    f"name {expected_source_id!r}"
                )
            try:
                frame_index = int(row["frame_index"])
                track_id = int(row["track_id"])
                coordinates = tuple(float(row[field]) for field in ("x1", "y1", "x2", "y2"))
            except ValueError as exc:
                raise ContractError(f"{context}: frame, track, or bounding box is invalid") from exc
            if frame_index < 0 or track_id < 0:
                raise ContractError(f"{context}: frame_index and track_id must be non-negative")
            if not all(math.isfinite(value) for value in coordinates):
                raise ContractError(f"{context}: bounding box must contain finite coordinates")
            x1, y1, x2, y2 = coordinates
            if x2 <= x1 or y2 <= y1:
                raise ContractError(f"{context}: bounding box must have positive width and height")
            key = (frame_index, track_id)
            if key in seen:
                raise ContractError(f"{context}: duplicate frame/track key {key}")
            seen.add(key)
            boxes.append(PreviewBox(frame_index, track_id, (x1, y1, x2, y2)))
    if not boxes:
        raise ContractError(f"Tracking manifest contains no track rows: {manifest_path}")
    return boxes


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable, bright BGR color without relying on global randomness."""
    return (
        64 + (track_id * 97) % 192,
        64 + (track_id * 57) % 192,
        64 + (track_id * 137) % 192,
    )


def _timestamp_text(frame_index: int, fps: float) -> str:
    total_ms = round(frame_index * 1000.0 / fps)
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"frame {frame_index:06d}  {minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def render_track_preview(
    video_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    source_id: str,
    overwrite: bool = False,
) -> int:
    """Render one complete normalized video and return its written frame count."""
    if not video_path.is_file():
        raise ContractError(f"Normalized video does not exist: {video_path}")
    if output_path.exists() and not overwrite:
        raise ContractError(
            f"Track preview already exists; pass --overwrite to replace it: {output_path}"
        )
    boxes = _load_preview_boxes(manifest_path, source_id)
    by_frame: dict[int, list[PreviewBox]] = defaultdict(list)
    for box in boxes:
        by_frame[box.frame_index].append(box)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ContractError(f"Could not open normalized video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ContractError(
            f"Normalized video has invalid FPS or geometry: {video_path} "
            f"(fps={fps}, size={width}x{height})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise ContractError(f"Could not create track preview: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for box in sorted(by_frame.get(frame_index, ()), key=lambda item: item.track_id):
                x1, y1, x2, y2 = box.bbox_xyxy
                left = max(0, min(width - 1, int(round(x1))))
                top = max(0, min(height - 1, int(round(y1))))
                right = max(0, min(width - 1, int(round(x2))))
                bottom = max(0, min(height - 1, int(round(y2))))
                if right <= left or bottom <= top:
                    continue
                color = _track_color(box.track_id)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"ID {box.track_id}"
                label_y = max(18, top - 6)
                cv2.putText(
                    frame,
                    label,
                    (left, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    label,
                    (left, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            timestamp = _timestamp_text(frame_index, fps)
            cv2.putText(
                frame,
                timestamp,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                timestamp,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    maximum_manifest_frame = max(by_frame)
    if frame_index == 0 or maximum_manifest_frame >= frame_index:
        output_path.unlink(missing_ok=True)
        raise ContractError(
            f"Manifest references frame {maximum_manifest_frame}, but normalized video "
            f"contains only {frame_index} readable frames: {video_path}"
        )
    return frame_index
