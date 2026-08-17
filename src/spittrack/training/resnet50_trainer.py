"""Two-stage ResNet50 training with resumable scientifically complete checkpoints."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from spittrack.evaluation.window_metrics import binary_window_metrics, select_f1_threshold
from spittrack.models.resnet50 import ResNet50Classifier

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochEvaluation:
    loss: float
    labels: list[int]
    probabilities: list[float]
    elapsed_s: float
    metadata: list[dict[str, Any]]


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=device.type == "cuda")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for clips, labels, _metadata in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = model(clips)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * labels.shape[0]
        total_count += labels.shape[0]
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochEvaluation:
    model.eval()
    start = time.perf_counter()
    total_loss = 0.0
    labels_out: list[int] = []
    probabilities: list[float] = []
    metadata_out: list[dict[str, Any]] = []
    for clips, labels, metadata in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device):
            logits = model(clips)
            loss = criterion(logits, labels)
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
        labels_out.extend(labels.cpu().tolist())
        probabilities.extend(probability.cpu().tolist())
        for index in range(labels.shape[0]):
            item: dict[str, Any] = {}
            for key, values in metadata.items():
                value = values[index]
                item[key] = value.item() if hasattr(value, "item") else value
            metadata_out.append(item)
        total_loss += float(loss) * labels.shape[0]
    return EpochEvaluation(
        loss=total_loss / max(len(labels_out), 1),
        labels=labels_out,
        probabilities=probabilities,
        elapsed_s=time.perf_counter() - start,
        metadata=metadata_out,
    )


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    stage: int,
    epoch: int,
    best_f1: float,
    best_epoch: int,
    threshold: float,
    config_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "stage": stage,
        "epoch": epoch,
        "best_validation_f1": best_f1,
        "best_epoch": best_epoch,
        "validation_threshold": threshold,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
    }


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def train_resnet50(
    model: ResNet50Classifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    *,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    config_sha256: str,
    manifest_sha256: str,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Execute protocol stages and save best.pt, last.pt, epoch log, and validation metrics."""
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    stages = [
        {
            "number": 1,
            "epochs": int(config["stage_1"]["epochs"]),
            "patience": None,
        },
        {
            "number": 2,
            "epochs": int(config["stage_2"]["max_epochs"]),
            "patience": int(config["stage_2"]["early_stopping_patience"]),
        },
    ]
    best_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    global_epoch = 0
    resume_stage = 1
    resume_epoch = 0
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_payload["config_sha256"] != config_sha256:
            raise ValueError("Resume checkpoint config hash differs from the committed run config")
        if resume_payload["manifest_sha256"] != manifest_sha256:
            raise ValueError("Resume checkpoint fold manifest hash differs from the current manifest")
        model.load_state_dict(resume_payload["model_state"])
        best_f1 = float(resume_payload["best_validation_f1"])
        best_epoch = int(resume_payload["best_epoch"])
        best_threshold = float(resume_payload["validation_threshold"])
        resume_stage = int(resume_payload["stage"])
        resume_epoch = int(resume_payload["epoch"])

    log_path = run_dir / "metrics.csv"
    if not log_path.exists():
        log_path.write_text(
            "global_epoch,stage,stage_epoch,train_loss,val_loss,val_f1,val_threshold,val_seconds\n",
            encoding="utf-8",
        )
    for stage in stages:
        stage_number = int(stage["number"])
        if stage_number < resume_stage:
            global_epoch += int(stage["epochs"])
            continue
        if stage_number == 1:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                model.backbone.fc.parameters(),
                lr=float(config["stage_1"]["lr_head"]),
                weight_decay=float(config["stage_2"]["weight_decay"]),
            )
        else:
            model.unfreeze_layer4_and_head()
            optimizer = torch.optim.AdamW(
                model.optimizer_groups(
                    backbone_lr=float(config["stage_2"]["lr_backbone"]),
                    head_lr=float(config["stage_2"]["lr_head"]),
                ),
                weight_decay=float(config["stage_2"]["weight_decay"]),
            )
        if resume_payload is not None and stage_number == resume_stage:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            scaler.load_state_dict(resume_payload["scaler_state"])
        stale_epochs = 0
        first_epoch = resume_epoch + 1 if stage_number == resume_stage else 1
        if stage_number == resume_stage:
            global_epoch += resume_epoch
        for stage_epoch in range(first_epoch, int(stage["epochs"]) + 1):
            global_epoch += 1
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            validation = evaluate_epoch(model, validation_loader, criterion, device)
            threshold, validation_f1 = select_f1_threshold(
                validation.labels, validation.probabilities
            )
            improved = validation_f1 > best_f1
            if improved:
                best_f1 = validation_f1
                best_epoch = global_epoch
                best_threshold = threshold
                stale_epochs = 0
            else:
                stale_epochs += 1
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                stage=stage_number,
                epoch=stage_epoch,
                best_f1=best_f1,
                best_epoch=best_epoch,
                threshold=best_threshold,
                config_sha256=config_sha256,
                manifest_sha256=manifest_sha256,
            )
            _atomic_torch_save(payload, run_dir / "last.pt")
            if improved:
                _atomic_torch_save(payload, run_dir / "best.pt")
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"{global_epoch},{stage_number},{stage_epoch},{train_loss:.8f},"
                    f"{validation.loss:.8f},{validation_f1:.8f},{threshold:.8f},"
                    f"{validation.elapsed_s:.6f}\n"
                )
            LOGGER.info(
                "stage=%d epoch=%d train_loss=%.5f val_loss=%.5f val_f1=%.5f threshold=%.5f",
                stage_number,
                stage_epoch,
                train_loss,
                validation.loss,
                validation_f1,
                threshold,
            )
            if stage["patience"] is not None and stale_epochs >= int(stage["patience"]):
                LOGGER.info("Early stopping stage 2 after %d stale epochs", stale_epochs)
                break
        resume_epoch = 0
        resume_payload = None
    best_payload = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    validation = evaluate_epoch(model, validation_loader, criterion, device)
    test = evaluate_epoch(model, test_loader, criterion, device)
    frozen_threshold = float(best_payload["validation_threshold"])
    metrics = binary_window_metrics(
        validation.labels,
        validation.probabilities,
        threshold=frozen_threshold,
    )
    test_metrics = binary_window_metrics(test.labels, test.probabilities, threshold=frozen_threshold)
    result = {
        "model": "resnet50",
        "best_epoch": int(best_payload["best_epoch"]),
        "validation_threshold": float(best_payload["validation_threshold"]),
        "validation": metrics,
        "outer_test": test_metrics,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictions_path = run_dir / "outer_test_predictions.csv"
    with predictions_path.open("w", encoding="utf-8") as stream:
        stream.write("window_id,source_id,track_id,label,probability,threshold,prediction\n")
        for metadata, label, probability in zip(test.metadata, test.labels, test.probabilities):
            stream.write(
                f"{metadata['window_id']},{metadata['source_id']},{metadata['track_id']},"
                f"{label},{probability:.10f},{frozen_threshold:.10f},"
                f"{int(probability >= frozen_threshold)}\n"
            )
    return result
