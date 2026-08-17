"""CPU-default offline full-video inference and evidence rendering."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import numpy as np
import torch

from spittrack.common.config import load_yaml
from spittrack.common.errors import ContractError
from spittrack.common.hashing import canonical_sha256
from spittrack.common.logging_utils import attach_run_log
from spittrack.common.reproducibility import (
    capture_package_freeze,
    capture_run_environment,
    seed_everything,
)
from spittrack.data.contracts import SourceRecord
from spittrack.data.extract_tracks import extract_track_manifest
from spittrack.data.materialize import materialize_window_clips
from spittrack.data.normalize import normalize_video
from spittrack.data.tracking import load_tracking_manifests
from spittrack.data.windows import build_window_index
from spittrack.inference.event_aggregator import WindowScore, aggregate_events
from spittrack.models.resnet50 import ResNet50Classifier
from spittrack.models.resnet_lstm import ResNet50LSTM
from spittrack.models.videomae import VideoMAEClassifier
from spittrack.training.dataset import transform_clip


def load_final_model(provenance_path: Path, *, device: torch.device):
    """Load an architecture without network access and return its frozen threshold."""
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = {"model_name", "checkpoint_path", "median_fold_validation_threshold", "config_sha256"}
    missing = sorted(required - set(provenance))
    if missing:
        raise ContractError(f"Final-model provenance is incomplete; missing {missing}")
    checkpoint_path = Path(str(provenance["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = provenance_path.parent / checkpoint_path
    if not checkpoint_path.is_file():
        raise ContractError(f"Final checkpoint is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != provenance["config_sha256"]:
        raise ContractError("Checkpoint/provenance config hashes differ")
    threshold = float(provenance["median_fold_validation_threshold"])
    if payload.get("frozen_threshold") != threshold:
        raise ContractError("Checkpoint/provenance frozen thresholds differ")
    name = str(provenance["model_name"])
    snapshot_path = Path(str(provenance.get("config_snapshot_path", "config_snapshot.json")))
    if not snapshot_path.is_absolute():
        snapshot_path = provenance_path.parent / snapshot_path
    if not snapshot_path.is_file():
        raise ContractError(f"Final config snapshot is missing: {snapshot_path}")
    complete_config = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if canonical_sha256(complete_config) != provenance["config_sha256"]:
        raise ContractError("Final config snapshot/provenance hashes differ")
    config = complete_config["model"]
    if name == "resnet50":
        model = ResNet50Classifier(pretrained=False, dropout=float(config.get("head_dropout", 0.3)))
    elif name == "resnet_lstm":
        values = config["lstm"]
        model = ResNet50LSTM(pretrained=False, hidden_size=int(values["hidden_size"]), num_layers=int(values["num_layers"]), dropout=float(values["dropout"]), classifier_dropout=float(config["classifier_dropout"]))
    elif name == "videomae":
        hf_config = payload.get("videomae_config")
        if not isinstance(hf_config, dict):
            raise ContractError("Offline VideoMAE inference requires videomae_config in final.pt")
        model = VideoMAEClassifier(final_trainable_blocks=int(config["initial_trainable_blocks"]), model_config=hf_config)
    else:
        raise ContractError(f"Unsupported final track-wise model {name!r}")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, threshold, provenance


def _write_events(
    events,
    output_dir: Path,
    *,
    video_file: str,
    model_name: str,
    checkpoint: str,
) -> None:
    rows = []
    for index, event in enumerate(events, start=1):
        event_id = f"event_{index:04d}"
        rows.append(
            {
                "source_id": event.source_id,
                "video_file": video_file,
                "event_id": event_id,
                "track_id": event.track_id,
                "event_type": "spitting",
                "start_s": event.start_s,
                "end_s": event.end_s,
                "duration_s": event.end_s - event.start_s,
                "confidence": event.confidence,
                "peak_x1": event.peak_bbox_xyxy[0],
                "peak_y1": event.peak_bbox_xyxy[1],
                "peak_x2": event.peak_bbox_xyxy[2],
                "peak_y2": event.peak_bbox_xyxy[3],
                "model_name": model_name,
                "checkpoint": checkpoint,
                "evidence_clip_path": f"evidence/{event_id}.mp4",
                "contributing_window_count": event.contributing_window_count,
            }
        )
    fields = [
        "source_id", "video_file", "event_id", "track_id", "event_type",
        "start_s", "end_s", "duration_s", "confidence", "peak_x1", "peak_y1",
        "peak_x2", "peak_y2", "model_name", "checkpoint", "evidence_clip_path",
        "contributing_window_count",
    ]
    with (output_dir / "events.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output_dir / "events.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _render(video_path: Path, track_frames, events, output_dir: Path, *, fps: int = 25) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for inference rendering") from exc
    by_frame: dict[int, list[Any]] = {}
    for item in track_frames: by_frame.setdefault(item.frame_index, []).append(item)
    capture = cv2.VideoCapture(str(video_path))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    annotated = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(annotated), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not capture.isOpened() or not writer.isOpened():
        raise ContractError("Could not open inference video or annotated output")
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok: break
            timestamp = frame_index / fps
            for track in by_frame.get(frame_index, []):
                active = [event for event in events if event.track_id == track.track_id and event.start_s <= timestamp <= event.end_s]
                color = (0, 0, 255) if active else (80, 200, 80)
                x1, y1, x2, y2 = map(int, track.bbox_xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {track.track_id}" + (f" SPITTING {active[0].confidence:.2f}" if active else "")
                cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            writer.write(frame); frame_index += 1
    finally:
        capture.release(); writer.release()
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    for index, event in enumerate(events, start=1):
        start_frame, end_frame = max(0, int((event.start_s - 0.5) * fps)), int((event.end_s + 0.5) * fps)
        source = cv2.VideoCapture(str(annotated)); source.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        clip = cv2.VideoWriter(str(evidence_dir / f"event_{index:04d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        try:
            for _ in range(start_frame, end_frame + 1):
                ok, frame = source.read()
                if not ok: break
                clip.write(frame)
        finally:
            source.release(); clip.release()


def run_video_inference(input_path: Path, provenance_path: Path, output_dir: Path, *, source_id: str | None = None, device_name: str = "cpu", data_config_path: Path = Path("configs/data.yaml"), detector_config_path: Path = Path("configs/detector.yaml"), tracker_config_path: Path = Path("configs/tracker.yaml"), inference_config_path: Path = Path("configs/inference.yaml"), batch_size: int = 4) -> dict[str, Any]:
    """Normalize, track, classify rolling person windows, aggregate, and render one video."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ContractError(f"Inference output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    attach_run_log(output_dir / "run.log")
    started = perf_counter()
    source_id = source_id or input_path.stem
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
        raise ContractError(
            "Inference source_id must contain only letters, digits, underscores, and hyphens"
        )
    logging.info("Starting full-video inference input=%s device=%s", input_path, device_name)
    data_config = load_yaml(data_config_path)
    detector = load_yaml(detector_config_path)
    tracker = load_yaml(tracker_config_path)
    inference_config = load_yaml(inference_config_path)
    detector["device"] = device_name
    seed = int(inference_config.get("seed", 42))
    seed_everything(seed)
    model_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    config_snapshot = {
        "data": data_config,
        "detector": detector,
        "tracker": tracker,
        "inference": inference_config,
        "runtime_device": device_name,
    }
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "final_model_provenance.json").write_text(
        json.dumps(model_provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = {
        **capture_run_environment(),
        "seed": seed,
        "config_sha256": canonical_sha256(config_snapshot),
        "model_provenance_sha256": canonical_sha256(model_provenance),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "pip_freeze.txt").write_text(
        capture_package_freeze() + "\n", encoding="utf-8"
    )
    normalized_root = output_dir / "intermediate" / "normalized"
    normalized_path = normalized_root / f"{source_id}.mp4"
    video_config = data_config["video"]
    fps = int(video_config["normalized_fps"])
    norm = normalize_video(input_path, normalized_path, target_fps=fps)
    manifest_dir = output_dir / "intermediate" / "tracking" / "manifests"
    manifest_path = manifest_dir / f"{source_id}.csv"
    run_id = extract_track_manifest(normalized_path, manifest_path, source_id=source_id, detector_config=detector, tracker_config=tracker, fps=fps)
    source = SourceRecord(source_id, input_path.name, "synthetic", False, "non_spitting", "normal_activity", norm.output_duration_s, norm.input_fps, float(fps), norm.width, norm.height, "inference", "0" * 64, {})
    track_frames = load_tracking_manifests(manifest_dir, [source])
    records, summary = build_window_index([source], track_frames, [], fps=fps, window_seconds=float(video_config["window_seconds"]), stride_seconds=float(video_config["stride_seconds"]), sampled_frames=int(video_config["sampled_frames"]))
    if not records:
        raise ContractError("No complete 2-second person-track windows were produced")
    clips_root = output_dir / "intermediate" / "clips"
    materialize_window_clips(records, track_frames, normalized_root=normalized_root, clips_root=clips_root, image_size=int(video_config["image_size"]), padding_fraction=float(tracker["roi_padding_fraction"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but unavailable")
    model, threshold, provenance = load_final_model(provenance_path, device=device)
    scores: list[WindowScore] = []
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            batch_records = records[offset:offset + batch_size]
            clips = []
            for record in batch_records:
                array = np.load(clips_root / record.source_id / f"{record.window_id}.npy", allow_pickle=False)
                clips.append(transform_clip(torch.from_numpy(array.copy()).permute(0, 3, 1, 2)))
            probabilities = torch.softmax(model(torch.stack(clips).to(device)), dim=1)[:, 1].cpu().tolist()
            for record, probability in zip(batch_records, probabilities):
                scores.append(WindowScore(record.source_id, record.track_id, record.start_s, record.end_s, float(probability), record.bbox_xyxy))
    settings = inference_config["event_aggregation"]
    events = aggregate_events(scores, threshold=threshold, min_consecutive=int(settings["min_consecutive_positive_windows"]), merge_gap_s=float(settings["merge_gap_seconds"]), expected_stride_s=float(settings["expected_window_stride_seconds"]))
    _render(normalized_path, track_frames, events, output_dir, fps=fps)
    _write_events(
        events,
        output_dir,
        video_file=input_path.name,
        model_name=str(provenance["model_name"]),
        checkpoint=str(provenance["checkpoint_path"]),
    )
    elapsed = perf_counter() - started
    result = {
        "model_name": provenance["model_name"],
        "frozen_threshold": threshold,
        "tracking_run_id": run_id,
        "window_count": summary.emitted_windows,
        "event_count": len(events),
        "annotated_video": str((output_dir / "annotated.mp4").resolve()),
        "end_to_end_performance": {
            "device": device_name,
            "scope": "raw_video_to_tracking_windows_classification_events_and_rendered_artifacts",
            "input_duration_s": norm.output_duration_s,
            "wall_time_s": elapsed,
            "input_video_seconds_per_wall_second": norm.output_duration_s / max(elapsed, 1e-12),
            "effective_input_frames_per_second": norm.output_frames / max(elapsed, 1e-12),
        },
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info(
        "Completed full-video inference windows=%d events=%d wall_time_s=%.3f effective_fps=%.3f",
        summary.emitted_windows,
        len(events),
        elapsed,
        norm.output_frames / max(elapsed, 1e-12),
    )
    return result
