from __future__ import annotations

import csv
import json
from pathlib import Path

from spittrack.data.contracts import ActionInterval
from spittrack.evaluation.run_events import evaluate_run_events


def test_run_event_evaluation_appends_real_computed_metrics(tmp_path: Path, source_factory) -> None:
    fieldnames = [
        "window_id", "source_id", "track_id", "start_s", "end_s",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "negative_subtype",
        "label", "probability", "threshold", "prediction",
    ]
    with (tmp_path / "outer_test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in enumerate((0.9, 0.9, 0.9)):
            writer.writerow(
                {
                    "window_id": f"w{index}", "source_id": "S01", "track_id": 4,
                    "start_s": index * 0.5, "end_s": index * 0.5 + 2,
                    "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 20,
                    "negative_subtype": "", "label": 1, "probability": probability,
                    "threshold": 0.5, "prediction": 1,
                }
            )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"model": "resnet50"}), encoding="utf-8"
    )
    sources = [source_factory("S01", True, duration_s=5.0)]
    annotations = [
        ActionInterval(
            "S01", "run", 4, "main_actor", "spitting", 0.0, 3.0, None, 1.0, ""
        )
    ]
    metrics = evaluate_run_events(
        tmp_path,
        sources=sources,
        annotations=annotations,
        test_source_ids={"S01"},
        model_name="resnet50",
        threshold=0.5,
        enable_track_attribution=True,
    )
    assert metrics["true_positives"] == 1
    assert (tmp_path / "events.csv").is_file()
    with (tmp_path / "events.csv").open(encoding="utf-8", newline="") as stream:
        event_row = next(csv.DictReader(stream))
    assert event_row["event_type"] == "spitting"
    assert event_row["duration_s"] == "3.0"
    assert event_row["checkpoint"] == "best.pt"
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "event_outer_test_raw_threshold" in payload
    assert payload["event_aggregation_ablation"]["aggregated_predicted_events"] == 1
