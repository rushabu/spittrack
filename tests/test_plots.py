from __future__ import annotations

import csv
from pathlib import Path

from spittrack.data.contracts import NEGATIVE_SUBTYPES
from spittrack.evaluation.cross_validation import AggregatedMetric
from spittrack.evaluation.plots import (
    plot_cross_validation_comparisons,
    plot_outer_fold_diagnostics,
)


def test_paper_plots_are_generated_from_outer_predictions(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "resnet50" / "fold_01"
    run.mkdir(parents=True)
    with (run / "outer_test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "probability", "prediction"])
        writer.writeheader()
        writer.writerows(
            [
                {"label": 0, "probability": 0.1, "prediction": 0},
                {"label": 0, "probability": 0.4, "prediction": 0},
                {"label": 1, "probability": 0.8, "prediction": 1},
                {"label": 1, "probability": 0.9, "prediction": 1},
            ]
        )
    aggregates = [
        AggregatedMetric("resnet50", "event_f1", 1, 0.8, 0.0),
        AggregatedMetric("resnet50", "raw_event_f1", 1, 0.6, 0.0),
    ]
    aggregates.extend(
        AggregatedMetric("resnet50", f"hard_negative_fp_{subtype}", 1, 0.0, 0.0)
        for subtype in NEGATIVE_SUBTYPES
    )
    plots = tmp_path / "plots"
    written = plot_outer_fold_diagnostics(
        tmp_path, models=["resnet50"], folds=[1], plots_dir=plots
    )
    written += plot_cross_validation_comparisons(aggregates, plots)
    assert {path.name for path in written} == {
        "confusion_matrix_resnet50.png",
        "precision_recall_resnet50.png",
        "model_comparison.png",
        "temporal_aggregation_ablation.png",
        "hard_negative_breakdown.png",
    }
    assert all(path.stat().st_size > 0 for path in written)
