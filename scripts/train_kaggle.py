"""Independent Kaggle entry point: one model and one fold per self-describing run directory."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from spittrack.common.config import load_yaml
from spittrack.common.reproducibility import initialize_run_directory, seed_everything
from spittrack.data.folds import load_fold_manifest
from spittrack.models.resnet50 import ResNet50Classifier
from spittrack.training.dataset import WindowClipDataset, source_class_balanced_sampler
from spittrack.training.resnet50_trainer import train_resnet50


MODELS = ("resnet50", "resnet_lstm", "videomae", "videomae_fullframe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--fold", required=True, help="Fold number or 'all'")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--window-index", type=Path, default=Path("data/processed/windows/windows.csv"))
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


def _train_resnet_fold(args: argparse.Namespace, config: dict, fold_number: int) -> None:
    manifest_path = args.splits_dir / f"fold_{fold_number:02d}.json"
    manifest = load_fold_manifest(manifest_path)
    if manifest.seed != int(config.get("seed", 42)):
        raise ValueError(
            f"Fold seed {manifest.seed} differs from config seed {config.get('seed', 42)}"
        )
    train_dataset = WindowClipDataset(
        args.window_index,
        args.windows_root,
        manifest,
        "train",
        training=True,
    )
    validation_dataset = WindowClipDataset(
        args.window_index,
        args.windows_root,
        manifest,
        "validation",
    )
    test_dataset = WindowClipDataset(
        args.window_index,
        args.windows_root,
        manifest,
        "test",
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
    model = ResNet50Classifier(pretrained=True, dropout=float(config.get("head_dropout", 0.3)))
    model.to(device)
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
            config=config,
            fold_manifest_path=manifest_path,
            seed=int(config.get("seed", 42)),
        )
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_resnet50(
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


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_yaml(args.config)
        if config.get("model") != args.model:
            raise ValueError(
                f"--model {args.model!r} conflicts with config model {config.get('model')!r}"
            )
        if args.model != "resnet50":
            raise NotImplementedError(
                f"{args.model} training is not implemented in the current milestone; no run was started"
            )
        seed_everything(int(config.get("seed", 42)))
        for fold_number in _fold_numbers(args.fold, args.splits_dir):
            _train_resnet_fold(args, config, fold_number)
    except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError, KeyError) as exc:
        logging.error("Training failed before a valid result could be reported: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
