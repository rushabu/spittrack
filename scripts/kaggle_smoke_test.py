"""Day-one Kaggle CUDA/dependency/data/checkpoint smoke test without opening outer-test data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader

from spittrack.common.config import load_yaml
from spittrack.common.reproducibility import initialize_run_directory, seed_everything
from spittrack.common.logging_utils import attach_run_log
from spittrack.data.folds import load_fold_manifest
from spittrack.models.resnet50 import ResNet50Classifier
from spittrack.training.dataset import WindowClipDataset, source_class_balanced_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--config", type=Path, default=Path("configs/models/resnet50.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--window-index", type=Path, default=Path("data/processed/windows/windows.csv")
    )
    parser.add_argument("--windows-root", type=Path, default=Path("data/processed/windows"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle GPU is not enabled; CUDA is required for this smoke gate")
        config = load_yaml(args.config)
        data_config = load_yaml(args.data_config)
        seed = int(config.get("seed", 42))
        seed_everything(seed)
        manifest_path = args.splits_dir / f"fold_{args.fold:02d}.json"
        manifest = load_fold_manifest(manifest_path)
        dataset = WindowClipDataset(
            args.window_index,
            args.windows_root,
            manifest,
            "train",
            training=True,
            augmentation=data_config["augmentation"],
        )
        batch_size = min(2, len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=source_class_balanced_sampler(dataset),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        clips, labels, metadata = next(iter(loader))
        if clips.shape[1:] != (16, 3, 224, 224):
            raise RuntimeError(f"Unexpected materialized batch shape: {tuple(clips.shape)}")
        smoke_config = {
            "model": config,
            "data": data_config,
            "smoke_test": True,
            "smoke_scope": "one_train_batch_no_validation_or_outer_test_opened",
        }
        output_root = args.output_root or (
            Path("/kaggle/working/outputs") if Path("/kaggle").exists() else Path("outputs")
        )
        run_dir = output_root / "smoke" / "resnet50" / f"fold_{args.fold:02d}"
        provenance = initialize_run_directory(
            run_dir,
            config=smoke_config,
            fold_manifest_path=manifest_path,
            seed=seed,
        )
        attach_run_log(run_dir / "run.log")
        model = ResNet50Classifier(pretrained=True, dropout=float(config["head_dropout"]))
        model.freeze_backbone()
        device = torch.device("cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.backbone.fc.parameters(), lr=float(config["stage_1"]["lr_head"])
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            logits = model(clips)
            loss = nn.CrossEntropyLoss()(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "completed_optimizer_steps": 1,
            "config_sha256": provenance["config_sha256"],
            "fold": args.fold,
        }
        checkpoint_path = run_dir / "smoke_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if restored["completed_optimizer_steps"] != 1:
            raise RuntimeError("Checkpoint reload verification failed")
        packages = {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "ultralytics", "transformers")
        }
        for opencv_distribution in ("opencv-python-headless", "opencv-python"):
            try:
                packages[opencv_distribution] = importlib.metadata.version(opencv_distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        else:
            raise RuntimeError("No OpenCV Python distribution is installed")
        report = {
            "status": "passed",
            "scientific_result": False,
            "outer_test_opened": False,
            "fold": args.fold,
            "train_batch_size": batch_size,
            "train_batch_source_ids": sorted(set(metadata["source_id"])),
            "loss_is_finite": bool(torch.isfinite(loss).item()),
            "optimizer_steps": 1,
            "checkpoint_reload_verified": True,
            "elapsed_step_seconds": elapsed,
            "cuda_device": torch.cuda.get_device_name(0),
            "packages": packages,
        }
        if not report["loss_is_finite"]:
            raise RuntimeError("Smoke loss is non-finite")
        freeze_text = (run_dir / "pip_freeze.txt").read_text(encoding="utf-8")
        if freeze_text.strip() == "unavailable":
            raise RuntimeError("pip freeze was unavailable; environment lock cannot be certified")
        lock_path = output_root / "requirements-lock.txt"
        if lock_path.exists() and lock_path.read_text(encoding="utf-8") != freeze_text:
            raise RuntimeError(
                f"A different smoke-certified environment lock already exists: {lock_path}"
            )
        lock_path.write_text(freeze_text, encoding="utf-8")
        report["environment_lock_path"] = str(lock_path)
        (run_dir / "smoke_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        logging.error("Kaggle smoke test failed: %s", exc)
        return 2
    logging.info("Kaggle smoke test passed: %s", run_dir / "smoke_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
