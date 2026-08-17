"""Convert outer-test window predictions into events and append event metrics to a run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from spittrack.data.contracts import ActionInterval, SourceRecord
from spittrack.evaluation.event_metrics import GroundTruthEvent, event_metrics
from spittrack.inference.event_aggregator import WindowScore, aggregate_events
from spittrack.inference.event_aggregator import SpittingEvent


def evaluate_run_events(
    run_dir: Path,
    *,
    sources: Iterable[SourceRecord],
    annotations: Iterable[ActionInterval],
    test_source_ids: set[str],
    model_name: str,
    threshold: float,
    enable_track_attribution: bool,
    min_consecutive: int = 2,
    merge_gap_s: float = 1.0,
    expected_stride_s: float = 0.5,
    temporal_iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Evaluate merged events using the already-frozen fold validation threshold."""
    prediction_path = run_dir / "outer_test_predictions.csv"
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    windows = [
        WindowScore(
            source_id=row["source_id"],
            track_id=int(row["track_id"]),
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            probability=float(row["probability"]),
            bbox_xyxy=tuple(float(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")),
        )
        for row in rows
    ]
    events = aggregate_events(
        windows,
        threshold=threshold,
        min_consecutive=min_consecutive,
        merge_gap_s=merge_gap_s,
        expected_stride_s=expected_stride_s,
    )
    raw_events = [
        SpittingEvent(
            source_id=item.source_id,
            track_id=item.track_id,
            start_s=item.start_s,
            end_s=item.end_s,
            confidence=item.probability,
            peak_bbox_xyxy=item.bbox_xyxy,
            contributing_window_count=1,
        )
        for item in windows
        if item.probability >= threshold
    ]
    source_map = {source.source_id: source for source in sources if source.source_id in test_source_ids}
    truth = [
        GroundTruthEvent(item.source_id, item.track_id, item.start_s, item.end_s)
        for item in annotations
        if item.source_id in test_source_ids and item.action_label == "spitting"
    ]
    no_spit_ids = {
        source.source_id for source in source_map.values() if not source.contains_spit
    }
    attribution_ids: set[str] | None = None
    if enable_track_attribution:
        tracks_by_source: dict[str, set[int]] = {}
        for row in rows:
            track_id = int(row["track_id"])
            if track_id >= 0:
                tracks_by_source.setdefault(row["source_id"], set()).add(track_id)
        attribution_ids = {
            source_id for source_id, track_ids in tracks_by_source.items() if len(track_ids) > 1
        }
    metrics = event_metrics(
        events,
        truth,
        source_durations_s={key: value.duration_s for key, value in source_map.items()},
        no_spit_source_ids=no_spit_ids,
        attribution_source_ids=attribution_ids,
        minimum_iou=temporal_iou_threshold,
    )
    raw_metrics = event_metrics(
        raw_events,
        truth,
        source_durations_s={key: value.duration_s for key, value in source_map.items()},
        no_spit_source_ids=no_spit_ids,
        attribution_source_ids=attribution_ids,
        minimum_iou=temporal_iou_threshold,
    )
    events_payload = []
    for index, event in enumerate(events, start=1):
        source = source_map[event.source_id]
        events_payload.append({
            "source_id": event.source_id,
            "video_file": source.file_name,
            "event_id": f"event_{index:04d}",
            "track_id": event.track_id,
            "event_type": "spitting",
            "start_s": event.start_s,
            "end_s": event.end_s,
            "duration_s": event.end_s - event.start_s,
            "confidence": event.confidence,
            "peak_bbox_xyxy": list(event.peak_bbox_xyxy),
            "model_name": model_name,
            "checkpoint": "best.pt",
            "evidence_clip_path": "",
        })
    (run_dir / "events.json").write_text(
        json.dumps(events_payload, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "events.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "source_id", "video_file", "event_id", "track_id", "event_type",
            "start_s", "end_s", "duration_s", "confidence", "peak_x1", "peak_y1",
            "peak_x2", "peak_y2", "model_name", "checkpoint", "evidence_clip_path",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for payload in events_payload:
            bbox = payload["peak_bbox_xyxy"]
            writer.writerow(
                {
                    **{key: payload[key] for key in fieldnames if key in payload},
                    "peak_x1": bbox[0], "peak_y1": bbox[1],
                    "peak_x2": bbox[2], "peak_y2": bbox[3],
                    "model_name": model_name,
                }
            )
    metrics_path = run_dir / "metrics.json"
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_metrics["event_outer_test"] = metrics
    run_metrics["event_outer_test_raw_threshold"] = raw_metrics
    run_metrics["event_aggregation_ablation"] = {
        "raw_event_f1": raw_metrics["f1"],
        "aggregated_event_f1": metrics["f1"],
        "raw_predicted_events": raw_metrics["predicted_events"],
        "aggregated_predicted_events": metrics["predicted_events"],
    }
    run_metrics["track_attribution_applicable"] = enable_track_attribution
    metrics_path.write_text(
        json.dumps(run_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics
