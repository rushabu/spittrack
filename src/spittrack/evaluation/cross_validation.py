"""Strict complete-matrix aggregation of independent model/fold run outputs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable

from spittrack.common.errors import ContractError
from spittrack.data.contracts import NEGATIVE_SUBTYPES

WINDOW_METRICS = ("precision", "recall", "f1", "balanced_accuracy", "pr_auc")
EVENT_REQUIRED_METRICS = (
    "precision",
    "recall",
    "f1",
    "false_positives_per_minute",
)
EVENT_OPTIONAL_METRICS = (
    "mean_start_error_s",
    "mean_end_error_s",
    "track_attribution_accuracy",
)


@dataclass(frozen=True)
class AggregatedMetric:
    model: str
    metric: str
    fold_count: int
    mean: float
    std: float


def select_final_model_policy(
    output_root: Path,
    *,
    candidate_models: Iterable[str],
    folds: Iterable[int],
) -> dict[str, object]:
    """Select by mean event F1 then lower FP/min and freeze median epoch/threshold policy."""
    fold_list = list(folds)
    candidates = list(candidate_models)
    if not candidates or not fold_list:
        raise ContractError("Final-model candidates and folds cannot be empty")
    summaries: list[dict[str, object]] = []
    for model in candidates:
        event_f1: list[float] = []
        fp_per_minute: list[float] = []
        epochs: list[int] = []
        thresholds: list[float] = []
        config_hashes: set[str] = set()
        for fold in fold_list:
            metrics, environment = _load_run_metrics(
                output_root / "runs" / model / f"fold_{fold:02d}"
            )
            event = metrics.get("event_outer_test", {})
            try:
                event_f1.append(float(event["f1"]))
                fp_per_minute.append(float(event["false_positives_per_minute"]))
                epochs.append(int(metrics["best_epoch"]))
                thresholds.append(float(metrics["validation_threshold"]))
                config_hashes.add(str(environment["config_sha256"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError(
                    f"Cannot derive final policy from {model}/fold_{fold:02d}: {exc}"
                ) from exc
        if len(config_hashes) != 1:
            raise ContractError(f"Config hashes differ across {model} folds")
        summaries.append(
            {
                "model": model,
                "mean_event_f1": mean(event_f1),
                "mean_false_positives_per_minute": mean(fp_per_minute),
                "median_cv_best_epoch": int(median(epochs)),
                "median_fold_validation_threshold": float(median(thresholds)),
                "config_sha256": next(iter(config_hashes)),
            }
        )
    winner = sorted(
        summaries,
        key=lambda item: (
            -float(item["mean_event_f1"]),
            float(item["mean_false_positives_per_minute"]),
            str(item["model"]),
        ),
    )[0]
    return {
        "selection_rule": "maximize_mean_event_f1_then_minimize_mean_false_positives_per_minute",
        "candidate_summaries": summaries,
        "selected_model": winner["model"],
        "median_cv_best_epoch": winner["median_cv_best_epoch"],
        "median_fold_validation_threshold": winner["median_fold_validation_threshold"],
        "selected_config_sha256": winner["config_sha256"],
        "final_training_uses_validation_set": False,
        "policy_frozen_before_final_training": True,
    }


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
            required = (
                "metrics.json",
                "environment.json",
                "fold_manifest.json",
                "best.pt",
                "last.pt",
                "outer_test_predictions.csv",
                "events.csv",
                "events.json",
            )
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
        event_outer = metrics.get("event_outer_test")
        if not isinstance(event_outer, dict):
            raise ContractError(f"{model}/fold_{fold:02d} has no event_outer_test metric object")
        for metric in EVENT_REQUIRED_METRICS:
            value = event_outer.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContractError(
                    f"{model}/fold_{fold:02d} missing finite event_outer_test.{metric}"
                )
            values.setdefault((model, f"event_{metric}"), []).append(float(value))
        raw_event_outer = metrics.get("event_outer_test_raw_threshold")
        if not isinstance(raw_event_outer, dict):
            raise ContractError(
                f"{model}/fold_{fold:02d} has no event_outer_test_raw_threshold object"
            )
        raw_f1 = raw_event_outer.get("f1")
        if not isinstance(raw_f1, (int, float)) or not math.isfinite(float(raw_f1)):
            raise ContractError(f"{model}/fold_{fold:02d} missing finite raw event F1")
        values.setdefault((model, "raw_event_f1"), []).append(float(raw_f1))
        for metric in EVENT_OPTIONAL_METRICS:
            value = event_outer.get(metric)
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ContractError(
                        f"{model}/fold_{fold:02d} has invalid event_outer_test.{metric}"
                    )
                values.setdefault((model, f"event_{metric}"), []).append(float(value))
        hard_negatives = metrics.get("hard_negative_false_positive_counts")
        if not isinstance(hard_negatives, dict):
            raise ContractError(
                f"{model}/fold_{fold:02d} has no hard_negative_false_positive_counts object"
            )
        for subtype in sorted(NEGATIVE_SUBTYPES):
            value = hard_negatives.get(subtype)
            if not isinstance(value, int) or value < 0:
                raise ContractError(
                    f"{model}/fold_{fold:02d} missing non-negative hard-negative count {subtype}"
                )
            values.setdefault((model, f"hard_negative_fp_{subtype}"), []).append(float(value))
        config_hash = environment.get("config_sha256")
        if not isinstance(config_hash, str):
            raise ContractError(f"{model}/fold_{fold:02d} has no config_sha256")
        config_hashes[model].add(config_hash)
    inconsistent = {model: hashes for model, hashes in config_hashes.items() if len(hashes) != 1}
    if inconsistent:
        raise ContractError(f"Config hash differs across folds for the same model: {inconsistent}")

    aggregates: list[AggregatedMetric] = []
    for model in model_list:
        metrics_for_model = list(WINDOW_METRICS) + [
            f"event_{metric}" for metric in EVENT_REQUIRED_METRICS
        ]
        metrics_for_model.append("raw_event_f1")
        metrics_for_model += [
            f"event_{metric}"
            for metric in EVENT_OPTIONAL_METRICS
            if (model, f"event_{metric}") in values
        ]
        metrics_for_model += [f"hard_negative_fp_{item}" for item in sorted(NEGATIVE_SUBTYPES)]
        if model != "videomae_fullframe" and (
            model,
            "event_track_attribution_accuracy",
        ) not in values:
            raise ContractError(
                f"{model} has no applicable multi-person track-attribution result across any fold"
            )
        for metric in metrics_for_model:
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
