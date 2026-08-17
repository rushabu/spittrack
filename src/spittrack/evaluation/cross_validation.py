"""Strict complete-matrix aggregation of independent model/fold run outputs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

from spittrack.common.errors import ContractError

WINDOW_METRICS = ("precision", "recall", "f1", "balanced_accuracy", "pr_auc")


@dataclass(frozen=True)
class AggregatedMetric:
    model: str
    metric: str
    fold_count: int
    mean: float
    std: float


def _load_run_metrics(run_dir: Path) -> tuple[dict, dict]:
    metrics_path = run_dir / "metrics.json"
    environment_path = run_dir / "environment.json"
    if not metrics_path.is_file() or not environment_path.is_file():
        missing = [str(path) for path in (metrics_path, environment_path) if not path.is_file()]
        raise ContractError(f"Run is not self-describing; missing files: {missing}")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"Run JSON is invalid in {run_dir}: {exc}") from exc
    return metrics, environment


def aggregate_complete_matrix(
    output_root: Path,
    *,
    models: Iterable[str],
    folds: Iterable[int],
) -> list[AggregatedMetric]:
    """Refuse a partial matrix, then compute sample standard deviation across outer folds."""
    model_list = list(models)
    fold_list = list(folds)
    if not model_list or not fold_list:
        raise ContractError("Expected model and fold sets cannot be empty")
    missing_runs: list[str] = []
    run_dirs: dict[tuple[str, int], Path] = {}
    for model in model_list:
        for fold in fold_list:
            run_dir = output_root / "runs" / model / f"fold_{fold:02d}"
            run_dirs[(model, fold)] = run_dir
            required = ("metrics.json", "environment.json", "fold_manifest.json", "best.pt", "last.pt")
            missing = [name for name in required if not (run_dir / name).is_file()]
            if missing:
                missing_runs.append(f"{model}/fold_{fold:02d}: {', '.join(missing)}")
    if missing_runs:
        raise ContractError(
            "Cross-validation matrix is incomplete; aggregation refused. Missing exactly:\n- "
            + "\n- ".join(missing_runs)
        )

    values: dict[tuple[str, str], list[float]] = {}
    config_hashes: dict[str, set[str]] = {model: set() for model in model_list}
    for (model, fold), run_dir in run_dirs.items():
        metrics, environment = _load_run_metrics(run_dir)
        if metrics.get("model") != model:
            raise ContractError(
                f"Run identity mismatch: directory says {model}, metrics says {metrics.get('model')}"
            )
        outer = metrics.get("outer_test")
        if not isinstance(outer, dict):
            raise ContractError(f"{model}/fold_{fold:02d} has no outer_test metric object")
        for metric in WINDOW_METRICS:
            value = outer.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContractError(f"{model}/fold_{fold:02d} missing finite outer_test.{metric}")
            values.setdefault((model, metric), []).append(float(value))
        config_hash = environment.get("config_sha256")
        if not isinstance(config_hash, str):
            raise ContractError(f"{model}/fold_{fold:02d} has no config_sha256")
        config_hashes[model].add(config_hash)
    inconsistent = {model: hashes for model, hashes in config_hashes.items() if len(hashes) != 1}
    if inconsistent:
        raise ContractError(f"Config hash differs across folds for the same model: {inconsistent}")

    aggregates: list[AggregatedMetric] = []
    for model in model_list:
        for metric in WINDOW_METRICS:
            metric_values = values[(model, metric)]
            aggregates.append(
                AggregatedMetric(
                    model=model,
                    metric=metric,
                    fold_count=len(metric_values),
                    mean=mean(metric_values),
                    std=stdev(metric_values) if len(metric_values) > 1 else 0.0,
                )
            )
    return aggregates


def write_cross_validation_summary(
    aggregates: Iterable[AggregatedMetric], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "metric", "fold_count", "mean", "std"])
        writer.writeheader()
        for item in aggregates:
            writer.writerow(
                {
                    "model": item.model,
                    "metric": item.metric,
                    "fold_count": item.fold_count,
                    "mean": f"{item.mean:.10f}",
                    "std": f"{item.std:.10f}",
                }
            )

