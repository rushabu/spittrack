"""Paper-facing plots generated exclusively from completed outer-fold predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from spittrack.data.contracts import NEGATIVE_SUBTYPES
from spittrack.evaluation.cross_validation import AggregatedMetric


def _plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - research environment dependency
        raise RuntimeError("matplotlib is required to generate paper plots") from exc
    return plt


def plot_outer_fold_diagnostics(
    output_root: Path,
    *,
    models: Iterable[str],
    folds: Iterable[int],
    plots_dir: Path,
) -> list[Path]:
    """Create one aggregate confusion matrix and PR curve per model from outer-test rows."""
    from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

    plt = _plotting()
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in models:
        labels: list[int] = []
        scores: list[float] = []
        predictions: list[int] = []
        for fold in folds:
            path = output_root / "runs" / model / f"fold_{fold:02d}" / "outer_test_predictions.csv"
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    labels.append(int(row["label"]))
                    scores.append(float(row["probability"]))
                    predictions.append(int(row["prediction"]))
        if not labels:
            raise ValueError(f"No outer-test predictions available for plotting model {model}")

        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        fig, axis = plt.subplots(figsize=(4.8, 4.2))
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_xticks([0, 1], labels=["non-spitting", "spitting"])
        axis.set_yticks([0, 1], labels=["non-spitting", "spitting"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Ground truth")
        axis.set_title(f"Outer-fold confusion matrix: {model}")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        destination = plots_dir / f"confusion_matrix_{model}.png"
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        written.append(destination)

        precision, recall, _thresholds = precision_recall_curve(labels, scores)
        average_precision = average_precision_score(labels, scores)
        fig, axis = plt.subplots(figsize=(5.2, 4.2))
        axis.plot(recall, precision, linewidth=2, label=f"AP={average_precision:.3f}")
        axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
        axis.set_title(f"Outer-fold precision-recall: {model}")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower left")
        fig.tight_layout()
        destination = plots_dir / f"precision_recall_{model}.png"
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        written.append(destination)
    return written


def plot_cross_validation_comparisons(
    aggregates: Iterable[AggregatedMetric], plots_dir: Path
) -> list[Path]:
    """Plot model event F1 and hard-negative false-positive breakdown with fold error bars."""
    plt = _plotting()
    items = list(aggregates)
    plots_dir.mkdir(parents=True, exist_ok=True)
    event_rows = [item for item in items if item.metric == "event_f1"]
    if not event_rows:
        raise ValueError("Cross-validation aggregates contain no event_f1 rows")
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    positions = np.arange(len(event_rows))
    axis.bar(
        positions,
        [item.mean for item in event_rows],
        yerr=[item.std for item in event_rows],
        capsize=4,
        color="#3572A5",
    )
    axis.set_xticks(positions, labels=[item.model for item in event_rows], rotation=20, ha="right")
    axis.set_ylabel("Event F1 (mean +/- fold SD)")
    axis.set_ylim(0, 1.02)
    axis.set_title("Outer-fold event performance")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    model_path = plots_dir / "model_comparison.png"
    fig.savefig(model_path, dpi=180)
    plt.close(fig)

    aggregate_map = {(item.model, item.metric): item for item in items}
    models = [item.model for item in event_rows]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    positions = np.arange(len(models))
    width = 0.36
    raw = [aggregate_map[(model, "raw_event_f1")] for model in models]
    aggregated = [aggregate_map[(model, "event_f1")] for model in models]
    axis.bar(
        positions - width / 2,
        [item.mean for item in raw],
        width,
        yerr=[item.std for item in raw],
        capsize=3,
        label="Raw threshold",
    )
    axis.bar(
        positions + width / 2,
        [item.mean for item in aggregated],
        width,
        yerr=[item.std for item in aggregated],
        capsize=3,
        label="Temporal aggregation",
    )
    axis.set_xticks(positions, labels=models, rotation=20, ha="right")
    axis.set_ylabel("Event F1 (mean +/- fold SD)")
    axis.set_ylim(0, 1.02)
    axis.set_title("Raw thresholding vs temporal aggregation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    ablation_path = plots_dir / "temporal_aggregation_ablation.png"
    fig.savefig(ablation_path, dpi=180)
    plt.close(fig)

    subtype_rows = {
        (item.model, item.metric.removeprefix("hard_negative_fp_")): item
        for item in items
        if item.metric.startswith("hard_negative_fp_")
    }
    width = 0.8 / max(1, len(models))
    positions = np.arange(len(NEGATIVE_SUBTYPES))
    fig, axis = plt.subplots(figsize=(12, 5.2))
    for model_index, model in enumerate(models):
        means = [subtype_rows[(model, subtype)].mean for subtype in sorted(NEGATIVE_SUBTYPES)]
        stds = [subtype_rows[(model, subtype)].std for subtype in sorted(NEGATIVE_SUBTYPES)]
        axis.bar(
            positions - 0.4 + width / 2 + model_index * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=model,
        )
    axis.set_xticks(positions, labels=sorted(NEGATIVE_SUBTYPES), rotation=35, ha="right")
    axis.set_ylabel("False-positive windows (mean +/- fold SD)")
    axis.set_title("Hard-negative false-positive breakdown")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    hard_negative_path = plots_dir / "hard_negative_breakdown.png"
    fig.savefig(hard_negative_path, dpi=180)
    plt.close(fig)
    return [model_path, ablation_path, hard_negative_path]
