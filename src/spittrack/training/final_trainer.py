"""Fixed-policy final training with no validation or test access."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from spittrack.common.reproducibility import capture_rng_state, restore_rng_state
from spittrack.training.resnet50_trainer import _atomic_torch_save, _autocast
from spittrack.training.temporal_trainer import _linear_warmup_decay


def train_fixed_policy(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    run_dir: Path,
    device: torch.device,
    epochs: int,
    frozen_threshold: float,
    config_sha256: str,
    policy_sha256: str,
    gradient_accumulation_steps: int = 1,
    warmup_fraction: float = 0.0,
    gradient_clip_norm: float = 1.0,
    resume_path: Path | None = None,
    epoch_start_callback: Callable[[nn.Module, torch.optim.Optimizer, int], None] | None = None,
    use_linear_schedule: bool = True,
) -> dict[str, Any]:
    """Train exactly the frozen epoch count and never construct an evaluation loader."""
    if epochs < 1 or gradient_accumulation_steps < 1:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = round(total_steps * warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _linear_warmup_decay(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ) if use_linear_schedule else 1.0,
    )
    start_epoch = 1
    losses: list[float] = []
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        for key, expected in (
            ("config_sha256", config_sha256),
            ("policy_sha256", policy_sha256),
            ("frozen_threshold", frozen_threshold),
            ("frozen_epoch_count", epochs),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"Final-training resume mismatch for {key}")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scaler.load_state_dict(payload["scaler_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["epoch"]) + 1
        losses = [float(value) for value in payload.get("training_losses", [])]
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])

    log_path = run_dir / "training.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,optimizer_steps\n", encoding="utf-8")
    for epoch in range(start_epoch, epochs + 1):
        if epoch_start_callback is not None:
            epoch_start_callback(model, optimizer, epoch)
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
                partial = final_group_size > 0 and batch_index >= len(train_loader) - final_group_size
                divisor = final_group_size if partial else gradient_accumulation_steps
                loss = raw_loss / divisor
            scaler.scale(loss).backward()
            boundary = (batch_index + 1) % gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if use_linear_schedule:
                    scheduler.step()
                optimizer_steps += 1
            total_loss += float(raw_loss.detach()) * labels.shape[0]
            total_count += labels.shape[0]
        epoch_loss = total_loss / max(1, total_count)
        losses.append(epoch_loss)
        payload = {
            "model_name": model_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "frozen_epoch_count": epochs,
            "frozen_threshold": frozen_threshold,
            "config_sha256": config_sha256,
            "policy_sha256": policy_sha256,
            "final_training_uses_validation_set": False,
            "training_losses": losses,
            "rng_state": capture_rng_state(),
        }
        if model_name == "videomae" and hasattr(model, "model"):
            payload["videomae_config"] = model.model.config.to_dict()
        _atomic_torch_save(payload, run_dir / "last.pt")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{epoch},{epoch_loss:.8f},{optimizer_steps}\n")
    if not (run_dir / "last.pt").is_file():
        raise RuntimeError("Final training produced no checkpoint")
    final_payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    _atomic_torch_save(final_payload, run_dir / "final.pt")
    result = {
        "model": model_name,
        "trained_epochs": epochs,
        "frozen_threshold": frozen_threshold,
        "final_training_uses_validation_set": False,
        "training_source_count": len(set(train_loader.dataset.source_ids)),
        "training_window_count": len(train_loader.dataset),
        "positive_window_count": sum(train_loader.dataset.labels),
        "negative_window_count": len(train_loader.dataset.labels) - sum(train_loader.dataset.labels),
        "training_losses": losses,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
