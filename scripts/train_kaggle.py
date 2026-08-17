"""Independent Kaggle entry point: one model and one fold per self-describing run directory."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from spittrack.common.config import load_yaml
from spittrack.common.logging_utils import attach_run_log
from spittrack.common.reproducibility import initialize_run_directory, seed_everything
from spittrack.data.folds import load_fold_manifest
from spittrack.data.contracts import load_action_annotations, load_source_metadata
from spittrack.evaluation.run_events import evaluate_run_events
from spittrack.models.resnet50 import ResNet50Classifier
from spittrack.models.resnet_lstm import ResNet50LSTM
from spittrack.models.videomae import VideoMAEClassifier
from spittrack.training.dataset import WindowClipDataset, source_class_balanced_sampler
from spittrack.training.resnet50_trainer import train_resnet50
from spittrack.training.temporal_trainer import train_temporal_model


MODELS = ("resnet50", "resnet_lstm", "videomae", "videomae_fullframe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--fold", required=True, help="Fold number or 'all'")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.yaml"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/tracker.yaml"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/source_metadata.csv"))
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations/action_intervals.csv")
    )
    parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/inference.yaml")
    )
    parser.add_argument("--window-index", type=Path, default=Path("data/processed/windows/windows.csv"))
    parser.add_argument(
        "--fullframe-window-index",
        type=Path,
        default=Path("data/processed/windows/windows_fullframe.csv"),
    )
    parser.add_argument("--windows-root", type=Path, default=Path("data/processed/windows"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve_output_root(requested: Path | None) -> Path:
    if requested is not None:
        return requested
    return Path("/kaggle/working/outputs") if Path("/kaggle").exists() else Path("outputs")


def _fold_numbers(value: str, splits_dir: Path) -> list[int]:
    if value == "all":
        paths = sorted(splits_dir.glob("fold_??.json"))
        if not paths:
            raise FileNotFoundError(f"No committed fold manifests found in {splits_dir}")
        return [int(path.stem.split("_")[1]) for path in paths]
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError("--fold must be an integer or 'all'") from exc
    if number < 1:
        raise ValueError("--fold must be positive")
    return [number]


def _complete_run_config(args: argparse.Namespace, model_config: dict) -> dict:
    """Hash and snapshot every scientific configuration that produced/evaluates a run."""
    return {
        "model": model_config,
        "data": load_yaml(args.data_config),
        "detector": load_yaml(args.detector_config),
        "tracker": load_yaml(args.tracker_config),
        "inference_and_evaluation": load_yaml(args.inference_config),
    }


def _train_resnet_fold(args: argparse.Namespace, config: dict, fold_number: int) -> None:
    manifest_path = args.splits_dir / f"fold_{fold_number:02d}.json"
    manifest = load_fold_manifest(manifest_path)
    if manifest.seed != int(config.get("seed", 42)):
        raise ValueError(
            f"Fold seed {manifest.seed} differs from config seed {config.get('seed', 42)}"
        )
    train_dataset, validation_dataset, test_dataset = _build_datasets(
        args, manifest, index_path=args.window_index
    )
    sampler = source_class_balanced_sampler(train_dataset)
    batch_size = int(config["stage_2"]["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; Kaggle GPU must be enabled")
    output_root = _resolve_output_root(args.output_root)
    run_dir = output_root / "runs" / args.model / f"fold_{fold_number:02d}"
    resume_path = run_dir / "last.pt" if args.resume else None
    if args.resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume requested but last.pt is missing: {resume_path}")
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    else:
        environment = initialize_run_directory(
            run_dir,
            config=_complete_run_config(args, config),
            fold_manifest_path=manifest_path,
            seed=int(config.get("seed", 42)),
        )
    attach_run_log(run_dir / "run.log")
    model = ResNet50Classifier(pretrained=True, dropout=float(config.get("head_dropout", 0.3)))
    model.to(device)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = train_resnet50(
        model,
        train_loader,
        validation_loader,
        test_loader,
        config=config,
        run_dir=run_dir,
        device=device,
        config_sha256=str(environment["config_sha256"]),
        manifest_sha256=str(manifest_payload["manifest_sha256"]),
        resume_path=resume_path,
    )
    _append_dataset_counts(
        run_dir, (train_dataset, validation_dataset, test_dataset)
    )
    _append_event_evaluation(
        args,
        manifest,
        run_dir,
        model_name=args.model,
        threshold=float(result["validation_threshold"]),
    )
    logging.info(
        "Completed model=%s fold=%d best_epoch=%s validation_threshold=%.6f",
        args.model,
        fold_number,
        result["best_epoch"],
        float(result["validation_threshold"]),
    )


def _append_event_evaluation(
    args: argparse.Namespace,
    manifest,
    run_dir: Path,
    *,
    model_name: str,
    threshold: float,
) -> None:
    sources = load_source_metadata(args.metadata, require_files=False)
    annotations = load_action_annotations(args.annotations, sources)
    inference_config = load_yaml(args.inference_config)
    event_config = inference_config["event_aggregation"]
    if event_config.get("smoothing") != "median" or int(
        event_config.get("smoothing_window_count", 0)
    ) != 3:
        raise ValueError("Increment two supports only the predeclared 3-window median smoother")
    evaluate_run_events(
        run_dir,
        sources=sources,
        annotations=annotations,
        test_source_ids=set(manifest.test),
        model_name=model_name,
        threshold=threshold,
        enable_track_attribution=model_name != "videomae_fullframe",
        min_consecutive=int(event_config["min_consecutive_positive_windows"]),
        merge_gap_s=float(event_config["merge_gap_seconds"]),
        expected_stride_s=float(event_config["expected_window_stride_seconds"]),
        temporal_iou_threshold=float(
            inference_config["evaluation"]["temporal_iou_threshold"]
        ),
    )


def _append_dataset_counts(
    run_dir: Path,
    datasets: tuple[WindowClipDataset, WindowClipDataset, WindowClipDataset],
) -> None:
    partitions = ("train", "validation", "outer_test")
    counts = {}
    for partition, dataset in zip(partitions, datasets):
        track_pairs = {
            (row["source_id"], int(row["track_id"]))
            for row in dataset.rows
            if int(row["track_id"]) >= 0
        }
        counts[partition] = {
            "source_count": len(set(dataset.source_ids)),
            "track_count": len(track_pairs),
            "usable_window_count": len(dataset),
            "positive_window_count": sum(dataset.labels),
            "negative_window_count": len(dataset.labels) - sum(dataset.labels),
        }
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["data_counts"] = counts
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_datasets(
    args: argparse.Namespace,
    manifest,
    *,
    index_path: Path,
) -> tuple[WindowClipDataset, WindowClipDataset, WindowClipDataset]:
    augmentation = load_yaml(args.data_config)["augmentation"]
    return (
        WindowClipDataset(
            index_path,
            args.windows_root,
            manifest,
            "train",
            training=True,
            augmentation=augmentation,
        ),
        WindowClipDataset(index_path, args.windows_root, manifest, "validation"),
        WindowClipDataset(index_path, args.windows_root, manifest, "test"),
    )


def _build_loaders(
    args: argparse.Namespace,
    datasets: tuple[WindowClipDataset, WindowClipDataset, WindowClipDataset],
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, validation_dataset, test_dataset = datasets
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=source_class_balanced_sampler(train_dataset),
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        ),
    )


def _make_temporal_model_and_optimizer(model_name: str, config: dict):
    if model_name == "resnet_lstm":
        lstm = config["lstm"]
        model = ResNet50LSTM(
            pretrained=True,
            hidden_size=int(lstm["hidden_size"]),
            num_layers=int(lstm["num_layers"]),
            dropout=float(lstm["dropout"]),
            classifier_dropout=float(config["classifier_dropout"]),
        )
        optimizer = torch.optim.AdamW(
            model.optimizer_groups(
                backbone_lr=float(config["lr_resnet_layer4"]),
                head_lr=float(config["lr_lstm_and_head"]),
            ),
            weight_decay=float(config["weight_decay"]),
        )
        return model, optimizer
    if model_name in {"videomae", "videomae_fullframe"}:
        if bool(config.get("enable_optional_full_unfreeze", False)):
            raise ValueError(
                "Optional full VideoMAE unfreeze is disabled in increment two until a predeclared "
                "stability rule is approved; keep enable_optional_full_unfreeze=false"
            )
        model = VideoMAEClassifier(
            checkpoint=str(config["checkpoint"]),
            final_trainable_blocks=int(config.get("initial_trainable_blocks", 4)),
        )
        optimizer = torch.optim.AdamW(
            model.optimizer_groups(
                backbone_lr=float(config["lr_backbone"]),
                head_lr=float(config["lr_head"]),
            ),
            weight_decay=float(config["weight_decay"]),
        )
        return model, optimizer
    raise ValueError(f"Unsupported temporal model {model_name!r}")


def _runtime_parameters(model_name: str, config: dict) -> tuple[int, int, float, float]:
    if model_name == "resnet_lstm":
        return (
            int(config["batch_size"]),
            int(config["gradient_accumulation_steps"]),
            float(config["warmup_fraction"]),
            float(config["gradient_clip_norm"]),
        )
    return (
        int(config["physical_batch_size"]),
        int(config["gradient_accumulation_steps"]),
        float(config["warmup_fraction"]),
        float(config["gradient_clip_norm"]),
    )


def _record_oom_adjustment(
    run_dir: Path,
    *,
    old_batch: int,
    new_batch: int,
    old_accumulation: int,
    new_accumulation: int,
) -> None:
    path = run_dir / "runtime_adjustments.json"
    adjustments = []
    if path.is_file():
        adjustments = json.loads(path.read_text(encoding="utf-8"))
    adjustments.append(
        {
            "reason": "cuda_out_of_memory",
            "old_physical_batch_size": old_batch,
            "new_physical_batch_size": new_batch,
            "old_gradient_accumulation_steps": old_accumulation,
            "new_gradient_accumulation_steps": new_accumulation,
            "effective_batch_size_preserved": old_batch * old_accumulation,
        }
    )
    path.write_text(json.dumps(adjustments, indent=2) + "\n", encoding="utf-8")


def _train_temporal_fold(args: argparse.Namespace, config: dict, fold_number: int) -> None:
    manifest_path = args.splits_dir / f"fold_{fold_number:02d}.json"
    manifest = load_fold_manifest(manifest_path)
    if manifest.seed != int(config.get("seed", 42)):
        raise ValueError("Fold seed differs from model config seed")
    index_path = (
        args.fullframe_window_index if args.model == "videomae_fullframe" else args.window_index
    )
    datasets = _build_datasets(args, manifest, index_path=index_path)
    batch_size, accumulation, warmup_fraction, clip_norm = _runtime_parameters(
        args.model, config
    )
    output_root = _resolve_output_root(args.output_root)
    run_dir = output_root / "runs" / args.model / f"fold_{fold_number:02d}"
    resume_path = run_dir / "last.pt" if args.resume else None
    if args.resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume requested but last.pt is missing: {resume_path}")
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        adjustment_path = run_dir / "runtime_adjustments.json"
        if adjustment_path.is_file():
            latest = json.loads(adjustment_path.read_text(encoding="utf-8"))[-1]
            batch_size = int(latest["new_physical_batch_size"])
            accumulation = int(latest["new_gradient_accumulation_steps"])
    else:
        environment = initialize_run_directory(
            run_dir,
            config=_complete_run_config(args, config),
            fold_manifest_path=manifest_path,
            seed=int(config.get("seed", 42)),
        )
    attach_run_log(run_dir / "run.log")
    model, optimizer = _make_temporal_model_and_optimizer(args.model, config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; Kaggle GPU must be enabled")
    manifest_hash = json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_sha256"]

    def run_attempt(current_batch: int, current_accumulation: int, checkpoint: Path | None):
        nonlocal model, optimizer
        loaders = _build_loaders(args, datasets, batch_size=current_batch)
        model.to(device)
        return train_temporal_model(
            model,
            *loaders,
            model_name=args.model,
            optimizer=optimizer,
            run_dir=run_dir,
            device=device,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["early_stopping_patience"]),
            gradient_accumulation_steps=current_accumulation,
            warmup_fraction=warmup_fraction,
            gradient_clip_norm=clip_norm,
            config_sha256=str(environment["config_sha256"]),
            manifest_sha256=str(manifest_hash),
            runtime_batch_size=current_batch,
            resume_path=checkpoint,
        )

    try:
        result = run_attempt(batch_size, accumulation, resume_path)
    except RuntimeError as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom or device.type != "cuda" or batch_size <= 1:
            raise
        new_batch = 1
        new_accumulation = accumulation * batch_size
        logging.warning(
            "CUDA OOM: retrying from last completed epoch with physical batch %d -> 1 and "
            "accumulation %d -> %d",
            batch_size,
            accumulation,
            new_accumulation,
        )
        _record_oom_adjustment(
            run_dir,
            old_batch=batch_size,
            new_batch=new_batch,
            old_accumulation=accumulation,
            new_accumulation=new_accumulation,
        )
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        model, optimizer = _make_temporal_model_and_optimizer(args.model, config)
        retry_checkpoint = run_dir / "last.pt"
        result = run_attempt(
            new_batch,
            new_accumulation,
            retry_checkpoint if retry_checkpoint.is_file() else None,
        )
    _append_dataset_counts(run_dir, datasets)
    _append_event_evaluation(
        args,
        manifest,
        run_dir,
        model_name=args.model,
        threshold=float(result["validation_threshold"]),
    )
    logging.info(
        "Completed model=%s fold=%d best_epoch=%s validation_threshold=%.6f",
        args.model,
        fold_number,
        result["best_epoch"],
        float(result["validation_threshold"]),
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_yaml(args.config)
        if config.get("model") != args.model:
            raise ValueError(
                f"--model {args.model!r} conflicts with config model {config.get('model')!r}"
            )
        seed_everything(int(config.get("seed", 42)))
        for fold_number in _fold_numbers(args.fold, args.splits_dir):
            if args.model == "resnet50":
                _train_resnet_fold(args, config, fold_number)
            else:
                _train_temporal_fold(args, config, fold_number)
    except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError, KeyError) as exc:
        logging.error("Training failed before a valid result could be reported: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
