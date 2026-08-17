"""Shared resumable trainer for ResNet50-LSTM and pretrained VideoMAE variants."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from spittrack.evaluation.window_metrics import (
    binary_window_metrics,
    hard_negative_false_positive_counts,
    select_f1_threshold,
)
from spittrack.training.resnet50_trainer import (
    _atomic_torch_save,
    _autocast,
    evaluate_epoch,
)
from spittrack.common.reproducibility import capture_rng_state, restore_rng_state

LOGGER = logging.getLogger(__name__)


def _linear_warmup_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    remaining = total_steps - step
    return max(0.0, float(remaining) / max(1, total_steps - warmup_steps))


def _write_predictions(path: Path, evaluation, threshold: float) -> None:
    fieldnames = [
        "window_id",
        "source_id",
        "track_id",
        "start_s",
        "end_s",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "negative_subtype",
        "label",
        "probability",
        "threshold",
        "prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for metadata, label, probability in zip(
            evaluation.metadata, evaluation.labels, evaluation.probabilities
        ):
            writer.writerow(
                {
                    **{key: metadata[key] for key in fieldnames[:10]},
                    "label": label,
                    "probability": f"{probability:.10f}",
                    "threshold": f"{threshold:.10f}",
                    "prediction": int(probability >= threshold),
                }
            )


def train_temporal_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    *,
    model_name: str,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    device: torch.device,
    max_epochs: int,
    patience: int,
    gradient_accumulation_steps: int,
    warmup_fraction: float,
    gradient_clip_norm: float,
    config_sha256: str,
    manifest_sha256: str,
    runtime_batch_size: int,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Train one temporal model without ever selecting a threshold on outer-test data."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_steps = max_epochs * optimizer_steps_per_epoch
    warmup_steps = round(total_steps * warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _linear_warmup_decay(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    start_epoch = 1
    best_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    stale_epochs = 0
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["config_sha256"] != config_sha256:
            raise ValueError("Resume checkpoint config hash differs from current config")
        if payload["manifest_sha256"] != manifest_sha256:
            raise ValueError("Resume checkpoint fold manifest hash differs from current manifest")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scaler.load_state_dict(payload["scaler_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_f1 = float(payload["best_validation_f1"])
        best_epoch = int(payload["best_epoch"])
        best_threshold = float(payload["validation_threshold"])
        stale_epochs = int(payload.get("stale_epochs", 0))
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        recorded_batch = int(payload.get("runtime_batch_size", runtime_batch_size))
        recorded_accum = int(
            payload.get("gradient_accumulation_steps", gradient_accumulation_steps)
        )
        if recorded_batch * recorded_accum != runtime_batch_size * gradient_accumulation_steps:
            raise ValueError(
                "Resume effective batch size differs from checkpoint; OOM fallback must preserve it"
            )

    log_path = run_dir / "metrics.csv"
    if not log_path.exists():
        log_path.write_text(
            "epoch,train_loss,val_loss,val_f1,val_threshold,val_seconds,optimizer_steps\n",
            encoding="utf-8",
        )
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_count = 0
        optimizer_steps = 0
        final_group_size = len(train_loader) % gradient_accumulation_steps
        for batch_index, (clips, labels, _metadata) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(device):
                logits = model(clips)
                raw_loss = criterion(logits, labels)
                in_final_partial_group = (
                    final_group_size > 0
                    and batch_index >= len(train_loader) - final_group_size
                )
                divisor = final_group_size if in_final_partial_group else gradient_accumulation_steps
                loss = raw_loss / divisor
            scaler.scale(loss).backward()
            is_boundary = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if is_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
            total_loss += float(raw_loss.detach()) * labels.shape[0]
            total_count += labels.shape[0]
        train_loss = total_loss / max(1, total_count)
        validation = evaluate_epoch(model, validation_loader, criterion, device)
        threshold, validation_f1 = select_f1_threshold(
            validation.labels, validation.probabilities
        )
        improved = validation_f1 > best_f1
        if improved:
            best_f1 = validation_f1
            best_epoch = epoch
            best_threshold = threshold
            stale_epochs = 0
        else:
            stale_epochs += 1
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_f1": best_f1,
            "best_epoch": best_epoch,
            "validation_threshold": best_threshold,
            "stale_epochs": stale_epochs,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "runtime_batch_size": runtime_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "rng_state": capture_rng_state(),
        }
        _atomic_torch_save(payload, run_dir / "last.pt")
        if improved:
            _atomic_torch_save(payload, run_dir / "best.pt")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{epoch},{train_loss:.8f},{validation.loss:.8f},{validation_f1:.8f},"
                f"{threshold:.8f},{validation.elapsed_s:.6f},{optimizer_steps}\n"
            )
        LOGGER.info(
            "model=%s epoch=%d train_loss=%.5f val_loss=%.5f val_f1=%.5f threshold=%.5f",
            model_name,
            epoch,
            train_loss,
            validation.loss,
            validation_f1,
            threshold,
        )
        if stale_epochs >= patience:
            LOGGER.info("Early stopping %s after %d stale epochs", model_name, stale_epochs)
            break
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    validation = evaluate_epoch(model, validation_loader, criterion, device)
    test = evaluate_epoch(model, test_loader, criterion, device)
    frozen_threshold = float(best["validation_threshold"])
    result = {
        "model": model_name,
        "best_epoch": int(best["best_epoch"]),
        "validation_threshold": frozen_threshold,
        "runtime_batch_size": runtime_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "validation": binary_window_metrics(
            validation.labels, validation.probabilities, threshold=frozen_threshold
        ),
        "outer_test": binary_window_metrics(
            test.labels, test.probabilities, threshold=frozen_threshold
        ),
        "hard_negative_false_positive_counts": hard_negative_false_positive_counts(
            test.labels,
            test.probabilities,
            [str(item["negative_subtype"]) for item in test.metadata],
            threshold=frozen_threshold,
        ),
        "classifier_performance": {
            "device": str(device),
            "scope": "materialized_clip_classifier_only_including_dataloader",
            "outer_test_windows_per_second": len(test.labels) / max(test.elapsed_s, 1e-12),
            "outer_test_sampled_frames_per_second": (
                16 * len(test.labels) / max(test.elapsed_s, 1e-12)
            ),
            "outer_test_mean_latency_ms_per_window": (
                1000 * test.elapsed_s / max(len(test.labels), 1)
            ),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_predictions(run_dir / "outer_test_predictions.csv", test, frozen_threshold)
    return result
