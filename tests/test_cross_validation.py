from __future__ import annotations

import json
from pathlib import Path

import pytest

from spittrack.common.errors import ContractError
from spittrack.evaluation.cross_validation import aggregate_complete_matrix


def create_run(root: Path, model: str, fold: int, *, config_hash: str = "same") -> None:
    run = root / "runs" / model / f"fold_{fold:02d}"
    run.mkdir(parents=True)
    metrics = {
        "model": model,
        "outer_test": {
            "precision": 0.5 + fold / 100,
            "recall": 0.6,
            "f1": 0.55,
            "balanced_accuracy": 0.65,
            "pr_auc": 0.7,
        },
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "environment.json").write_text(
        json.dumps({"config_sha256": config_hash}), encoding="utf-8"
    )
    (run / "fold_manifest.json").write_text("{}", encoding="utf-8")
    (run / "best.pt").write_bytes(b"checkpoint")
    (run / "last.pt").write_bytes(b"checkpoint")


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

