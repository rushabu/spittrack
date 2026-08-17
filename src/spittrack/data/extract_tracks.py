"""Ultralytics YOLOv8s + ByteTrack extraction into immutable per-frame manifests."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from spittrack.common.hashing import canonical_sha256, file_sha256
from spittrack.data.boxes import bbox_height_at_720p

LOGGER = logging.getLogger(__name__)


def tracking_run_id(detector_config: dict[str, Any], tracker_config: dict[str, Any]) -> str:
    """Bind annotations to detector/tracker configuration and local weight bytes when available."""
    weights = Path(str(detector_config["weights"]))
    payload = {
        "detector": detector_config,
        "tracker": tracker_config,
        "weights_sha256": file_sha256(weights) if weights.is_file() else None,
    }
    return f"tracking-{canonical_sha256(payload)[:16]}"


def extract_track_manifest(
    video_path: Path,
    output_csv: Path,
    *,
    source_id: str,
    detector_config: dict[str, Any],
    tracker_config: dict[str, Any],
    fps: int = 25,
) -> str:
    """Run tracking over one normalized source and write one row per tracked person/frame."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised on Kaggle
        raise RuntimeError("ultralytics is required for person detection and tracking") from exc
    model = YOLO(detector_config["weights"])
    run_id = tracking_run_id(detector_config, tracker_config)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    results = model.track(
        source=str(video_path),
        stream=True,
        persist=bool(tracker_config.get("persist_tracks", True)),
        tracker=str(tracker_config.get("config", "bytetrack.yaml")),
        classes=list(detector_config.get("classes", [0])),
        imgsz=int(detector_config.get("imgsz", 640)),
        conf=float(detector_config.get("conf", 0.25)),
        iou=float(detector_config.get("iou", 0.5)),
        device=str(detector_config.get("device", "cpu")),
        verbose=False,
    )
    rows: list[dict[str, Any]] = []
    for frame_index, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.detach().cpu().tolist()
        coords = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        frame_height = int(result.orig_shape[0])
        for track_id, xyxy, confidence in zip(ids, coords, confidences):
            rows.append(
                {
                    "source_id": source_id,
                    "tracking_run_id": run_id,
                    "frame_index": frame_index,
                    "timestamp_s": f"{frame_index / fps:.6f}",
                    "track_id": int(track_id),
                    "x1": f"{xyxy[0]:.4f}",
                    "y1": f"{xyxy[1]:.4f}",
                    "x2": f"{xyxy[2]:.4f}",
                    "y2": f"{xyxy[3]:.4f}",
                    "det_conf": f"{confidence:.6f}",
                    "_frame_height": frame_height,
                }
            )
    if not rows:
        raise RuntimeError(f"Tracking produced no person trajectories for source {source_id}")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["track_id"])].append(row)
    accepted_ids: set[int] = set()
    rejections: list[dict[str, Any]] = []
    minimum_duration = float(tracker_config.get("minimum_track_duration_s", 2.0))
    minimum_height = float(tracker_config.get("minimum_bbox_height_px_at_720p", 80))
    for track_id, track_rows in sorted(grouped.items()):
        indices = [int(row["frame_index"]) for row in track_rows]
        duration_s = (max(indices) - min(indices) + 1) / fps
        normalized_heights = [
            bbox_height_at_720p(
                float(row["y2"]) - float(row["y1"]), int(row["_frame_height"])
            )
            for row in track_rows
        ]
        median_height = median(normalized_heights)
        reasons: list[str] = []
        if duration_s < minimum_duration:
            reasons.append("track_too_short")
        if median_height < minimum_height:
            reasons.append("bbox_too_small")
        if reasons:
            rejections.append(
                {
                    "source_id": source_id,
                    "tracking_run_id": run_id,
                    "track_id": track_id,
                    "duration_s": f"{duration_s:.6f}",
                    "median_bbox_height_at_720p": f"{median_height:.4f}",
                    "reason": "+".join(reasons),
                }
            )
            LOGGER.warning(
                "Rejected source=%s track=%d reasons=%s duration=%.3fs median_height_720p=%.2f",
                source_id,
                track_id,
                reasons,
                duration_s,
                median_height,
            )
        else:
            accepted_ids.add(track_id)
    accepted_rows = [row for row in rows if int(row["track_id"]) in accepted_ids]
    rejection_path = output_csv.parent.parent / "rejections" / output_csv.name
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    with rejection_path.open("w", encoding="utf-8", newline="") as stream:
        rejection_fields = [
            "source_id",
            "tracking_run_id",
            "track_id",
            "duration_s",
            "median_bbox_height_at_720p",
            "reason",
        ]
        writer = csv.DictWriter(stream, fieldnames=rejection_fields)
        writer.writeheader()
        writer.writerows(rejections)
    if not accepted_rows:
        raise RuntimeError(
            f"All {len(grouped)} trajectories for source {source_id} were rejected; see {rejection_path}"
        )
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(accepted_rows)
    LOGGER.info(
        "source=%s tracking_run_id=%s accepted_tracks=%d rejected_tracks=%d rows=%d",
        source_id,
        run_id,
        len(accepted_ids),
        len(rejections),
        len(accepted_rows),
    )
    return run_id
