"""Materialize deterministic padded track crops as uint8 NumPy clip arrays."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from spittrack.common.errors import ContractError, WindowError
from spittrack.data.boxes import padded_clamped_box
from spittrack.data.tracking import TrackFrame
from spittrack.data.windows import WindowRecord

LOGGER = logging.getLogger(__name__)


def materialize_window_clips(
    records: Iterable[WindowRecord],
    track_frames: Iterable[TrackFrame],
    *,
    normalized_root: Path,
    clips_root: Path,
    image_size: int = 224,
    padding_fraction: float = 0.18,
) -> list[Path]:
    """Decode each source once and save each sampled track clip as `[T,H,W,RGB]` uint8."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("opencv-python-headless is required to materialize track clips") from exc
    if image_size <= 0:
        raise WindowError("image_size must be positive")
    record_groups: dict[str, list[WindowRecord]] = defaultdict(list)
    for record in records:
        record_groups[record.source_id].append(record)
    boxes = {
        (frame.source_id, frame.track_id, frame.frame_index): frame.bbox_xyxy
        for frame in track_frames
    }
    written: list[Path] = []
    for source_id in sorted(record_groups):
        source_records = record_groups[source_id]
        video_path = normalized_root / f"{source_id}.mp4"
        if not video_path.is_file():
            raise ContractError(f"Normalized video does not exist for materialization: {video_path}")
        references: dict[int, list[tuple[int, int]]] = defaultdict(list)
        clip_arrays = [
            np.empty((len(record.sampled_frame_indices), image_size, image_size, 3), dtype=np.uint8)
            for record in source_records
        ]
        for record_index, record in enumerate(source_records):
            for position, frame_index in enumerate(record.sampled_frame_indices):
                references[frame_index].append((record_index, position))
        maximum_frame = max(references)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ContractError(f"Could not open normalized video: {video_path}")
        try:
            for frame_index in range(maximum_frame + 1):
                ok, frame_bgr = capture.read()
                if not ok:
                    raise ContractError(
                        f"Normalized video {source_id} ended before required frame {frame_index}"
                    )
                for record_index, position in references.get(frame_index, []):
                    record = source_records[record_index]
                    box_key = (source_id, record.track_id, frame_index)
                    if box_key not in boxes:
                        raise WindowError(f"Missing bbox while materializing {record.window_id}: {box_key}")
                    height, width = frame_bgr.shape[:2]
                    x1, y1, x2, y2 = padded_clamped_box(
                        boxes[box_key],
                        frame_width=width,
                        frame_height=height,
                        padding_fraction=padding_fraction,
                    )
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        raise WindowError(f"Zero-size crop while materializing {record.window_id}")
                    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
                    clip_arrays[record_index][position] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        for record, array in zip(source_records, clip_arrays):
            output = clips_root / source_id / f"{record.window_id}.npy"
            if output.exists():
                raise ContractError(f"Refusing to overwrite materialized clip: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, array, allow_pickle=False)
            written.append(output)
        LOGGER.info("source=%s materialized_windows=%d", source_id, len(source_records))
    return written
