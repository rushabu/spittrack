from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

import spittrack.inference.pipeline as pipeline
from spittrack.common.hashing import canonical_sha256


class AlwaysPositive(nn.Module):
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 5.0]], device=clips.device).repeat(clips.shape[0], 1)


class PortableModel(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return torch.stack((torch.zeros(clips.shape[0]), self.weight.repeat(clips.shape[0])), dim=1)


def _dummy_video(path: Path) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (96, 96))
    assert writer.isOpened()
    for index in range(75):
        frame = np.full((96, 96, 3), index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_cpu_full_video_inference_with_mocked_detector_and_classifier(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.mp4"
    _dummy_video(input_path)

    def fake_extract(video_path, output_csv, *, source_id, detector_config, tracker_config, fps):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["source_id", "tracking_run_id", "frame_index", "timestamp_s", "track_id", "x1", "y1", "x2", "y2", "det_conf"]
        with output_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for frame in range(75):
                writer.writerow({"source_id": source_id, "tracking_run_id": "mock-run", "frame_index": frame, "timestamp_s": frame / fps, "track_id": 7, "x1": 10, "y1": 10, "x2": 80, "y2": 90, "det_conf": 0.9})
        return "mock-run"

    monkeypatch.setattr(pipeline, "extract_track_manifest", fake_extract)
    monkeypatch.setattr(pipeline, "load_final_model", lambda path, device: (AlwaysPositive().to(device).eval(), 0.5, {"model_name": "mock"}))
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}", encoding="utf-8")
    output = tmp_path / "result"
    result = pipeline.run_video_inference(input_path, provenance, output, device_name="cpu", batch_size=2)
    assert result["tracking_run_id"] == "mock-run"
    assert result["event_count"] == 1
    assert (output / "annotated.mp4").stat().st_size > 0
    assert (output / "events.csv").is_file()
    assert (output / "events.json").is_file()
    assert list((output / "evidence").glob("*.mp4"))


def test_final_model_bundle_uses_portable_relative_paths(tmp_path: Path, monkeypatch) -> None:
    complete_config = {"model": {"model": "resnet50", "head_dropout": 0.3}}
    config_hash = canonical_sha256(complete_config)
    (tmp_path / "config_snapshot.json").write_text(__import__("json").dumps(complete_config), encoding="utf-8")
    torch.save({"model_state": PortableModel().state_dict(), "config_sha256": config_hash, "frozen_threshold": 0.61}, tmp_path / "final.pt")
    provenance = {"model_name": "resnet50", "checkpoint_path": "final.pt", "config_snapshot_path": "config_snapshot.json", "median_fold_validation_threshold": 0.61, "config_sha256": config_hash}
    provenance_path = tmp_path / "final_model_provenance.json"
    provenance_path.write_text(__import__("json").dumps(provenance), encoding="utf-8")
    monkeypatch.setattr(pipeline, "ResNet50Classifier", PortableModel)
    model, threshold, loaded = pipeline.load_final_model(provenance_path, device=torch.device("cpu"))
    assert isinstance(model, PortableModel)
    assert threshold == 0.61
    assert loaded["checkpoint_path"] == "final.pt"
