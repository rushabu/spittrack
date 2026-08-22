from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from spittrack.common.errors import ContractError
from spittrack.data.contracts import NEGATIVE_SUBTYPES
from spittrack.evaluation.cross_validation import (
    aggregate_complete_matrix,
    select_final_model_policy,
    write_partition_count_summary,
)


def create_run(root: Path, model: str, fold: int, *, config_hash: str = "same") -> None:
    run = root / "runs" / model / f"fold_{fold:02d}"
    run.mkdir(parents=True)
    metrics = {
        "model": model,
        "best_epoch": 4 + fold,
        "validation_threshold": 0.5 + fold / 100,
        "outer_test": {
            "precision": 0.5 + fold / 100,
            "recall": 0.6,
            "f1": 0.55,
            "balanced_accuracy": 0.65,
            "pr_auc": 0.7,
        },
        "event_outer_test": {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "false_positives_per_minute": 0.2,
            "mean_start_error_s": 0.1,
            "mean_end_error_s": 0.2,
            "track_attribution_accuracy": 1.0,
        },
        "event_outer_test_raw_threshold": {"f1": 0.4},
        "hard_negative_false_positive_counts": {
            subtype: 0 for subtype in NEGATIVE_SUBTYPES
        },
        "data_counts": {
            "train": {
                "source_count": 22,
                "family_count": 8,
                "track_count": 30,
                "usable_window_count": 100,
            },
            "validation": {
                "source_count": 7,
                "family_count": 2,
                "track_count": 9,
                "usable_window_count": 25,
            },
            "outer_test": {
                "source_count": 8,
                "family_count": 2,
                "track_count": 10,
                "usable_window_count": 30,
            },
        },
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "environment.json").write_text(
        json.dumps({"config_sha256": config_hash}), encoding="utf-8"
    )
    (run / "fold_manifest.json").write_text("{}", encoding="utf-8")
    (run / "best.pt").write_bytes(b"checkpoint")
    (run / "last.pt").write_bytes(b"checkpoint")
    (run / "outer_test_predictions.csv").write_text("window_id\n", encoding="utf-8")
    (run / "events.csv").write_text("source_id\n", encoding="utf-8")
    (run / "events.json").write_text("[]", encoding="utf-8")
    (run / "run.log").write_text("completed\n", encoding="utf-8")


def test_partial_matrix_is_refused_with_exact_missing_run(tmp_path: Path) -> None:
    create_run(tmp_path, "resnet50", 1)
    with pytest.raises(ContractError, match="resnet50/fold_02"):
        aggregate_complete_matrix(tmp_path, models=["resnet50"], folds=[1, 2])


def test_complete_matrix_aggregates_mean_and_sample_std(tmp_path: Path) -> None:
    create_run(tmp_path, "resnet50", 1)
    create_run(tmp_path, "resnet50", 2)
    results = aggregate_complete_matrix(tmp_path, models=["resnet50"], folds=[1, 2])
    precision = next(item for item in results if item.metric == "precision")
    assert precision.fold_count == 2
    assert precision.mean == pytest.approx(0.515)
    assert precision.std > 0


def test_config_hash_mismatch_across_folds_is_refused(tmp_path: Path) -> None:
    create_run(tmp_path, "resnet50", 1, config_hash="a")
    create_run(tmp_path, "resnet50", 2, config_hash="b")
    with pytest.raises(ContractError, match="Config hash differs"):
        aggregate_complete_matrix(tmp_path, models=["resnet50"], folds=[1, 2])


def test_cv_partition_table_reports_sources_and_families(tmp_path: Path) -> None:
    create_run(tmp_path, "resnet50", 1)
    destination = tmp_path / "paper" / "partition_counts.csv"
    write_partition_count_summary(
        tmp_path,
        models=["resnet50"],
        folds=[1],
        output_path=destination,
    )
    with destination.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["source_video_count"] == "22"
    assert rows[0]["family_count"] == "8"
    assert rows[0]["source_family_summary"] == "22 sources / 8 families"


def test_cv_aggregation_refuses_missing_family_count(tmp_path: Path) -> None:
    create_run(tmp_path, "resnet50", 1)
    path = tmp_path / "runs" / "resnet50" / "fold_01" / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["data_counts"]["outer_test"]["family_count"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="outer_test.family_count"):
        aggregate_complete_matrix(tmp_path, models=["resnet50"], folds=[1])


def test_final_policy_uses_event_f1_then_fp_and_median_cv_values(tmp_path: Path) -> None:
    create_run(tmp_path, "resnet50", 1)
    create_run(tmp_path, "resnet50", 2)
    create_run(tmp_path, "videomae", 1)
    create_run(tmp_path, "videomae", 2)
    for fold in (1, 2):
        path = tmp_path / "runs" / "videomae" / f"fold_{fold:02d}" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["event_outer_test"]["f1"] = 0.8
        path.write_text(json.dumps(payload), encoding="utf-8")
    policy = select_final_model_policy(
        tmp_path, candidate_models=["resnet50", "videomae"], folds=[1, 2]
    )
    assert policy["selected_model"] == "videomae"
    assert policy["median_cv_best_epoch"] == 5
    assert policy["final_training_uses_validation_set"] is False
