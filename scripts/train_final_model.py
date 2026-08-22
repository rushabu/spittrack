"""Train the selected final model on all synthetic sources under a frozen CV policy."""

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
from spittrack.common.errors import ContractError
from spittrack.common.hashing import canonical_sha256
from spittrack.common.logging_utils import attach_run_log
from spittrack.common.reproducibility import initialize_run_directory, seed_everything
from spittrack.data.contracts import load_source_metadata
from spittrack.data.folds import FoldManifest
from spittrack.models.resnet50 import ResNet50Classifier
from spittrack.models.resnet_lstm import ResNet50LSTM
from spittrack.models.videomae import VideoMAEClassifier
from spittrack.training.dataset import WindowClipDataset, source_class_balanced_sampler
from spittrack.training.final_trainer import train_fixed_policy

MODEL_CONFIGS = {
    "resnet50": Path("configs/models/resnet50.yaml"),
    "resnet_lstm": Path("configs/models/resnet_lstm.yaml"),
    "videomae": Path("configs/models/videomae.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("outputs/cross_validation/final_model_policy.json"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/source_metadata.csv"))
    parser.add_argument("--window-index", type=Path, default=Path("data/processed/windows/windows.csv"))
    parser.add_argument("--windows-root", type=Path, default=Path("data/processed/windows"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.yaml"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/tracker.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_frozen_policy(path: Path) -> dict:
    if not path.is_file():
        raise ContractError(f"Frozen final-model policy is missing: {path}")
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {"selected_model", "median_cv_best_epoch", "median_fold_validation_threshold", "selected_config_sha256", "cv_summary_sha256", "final_training_uses_validation_set", "policy_frozen_before_final_training"}
    missing = sorted(required - set(policy))
    if missing:
        raise ContractError(f"Final-model policy is incomplete; missing {missing}")
    if policy["selected_model"] not in MODEL_CONFIGS:
        raise ContractError("Final policy must select a track-wise model; full-frame attribution is inapplicable")
    if policy["final_training_uses_validation_set"] is not False or policy["policy_frozen_before_final_training"] is not True:
        raise ContractError("Final policy must be frozen and declare no validation set")
    if int(policy["median_cv_best_epoch"]) < 1:
        raise ContractError("Frozen median CV best epoch must be positive")
    threshold = float(policy["median_fold_validation_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ContractError("Frozen median validation threshold must be in [0,1]")
    return policy


def _complete_config(args: argparse.Namespace, model_config: dict) -> dict:
    return {
        "model": model_config,
        "data": load_yaml(args.data_config),
        "detector": load_yaml(args.detector_config),
        "tracker": load_yaml(args.tracker_config),
        "inference_and_evaluation": load_yaml(args.inference_config),
    }


def _model_optimizer(name: str, config: dict):
    if name == "resnet50":
        model = ResNet50Classifier(pretrained=True, dropout=float(config.get("head_dropout", 0.3)))
        model.unfreeze_layer4_and_head()
        optimizer = torch.optim.AdamW(
            model.optimizer_groups(
                backbone_lr=float(config["stage_2"]["lr_backbone"]),
                head_lr=float(config["stage_2"]["lr_head"]),
            ), weight_decay=float(config["stage_2"]["weight_decay"])
        )
        def stage_callback(current_model, current_optimizer, epoch):
            if epoch <= int(config["stage_1"]["epochs"]):
                current_model.freeze_backbone()
                current_optimizer.param_groups[0]["lr"] = 0.0
                current_optimizer.param_groups[1]["lr"] = float(config["stage_1"]["lr_head"])
            else:
                current_model.unfreeze_layer4_and_head()
                current_optimizer.param_groups[0]["lr"] = float(config["stage_2"]["lr_backbone"])
                current_optimizer.param_groups[1]["lr"] = float(config["stage_2"]["lr_head"])
        return model, optimizer, 1, 0.0, 1.0, stage_callback, False
    if name == "resnet_lstm":
        lstm = config["lstm"]
        model = ResNet50LSTM(pretrained=True, hidden_size=int(lstm["hidden_size"]), num_layers=int(lstm["num_layers"]), dropout=float(lstm["dropout"]), classifier_dropout=float(config["classifier_dropout"]))
        optimizer = torch.optim.AdamW(model.optimizer_groups(backbone_lr=float(config["lr_resnet_layer4"]), head_lr=float(config["lr_lstm_and_head"])), weight_decay=float(config["weight_decay"]))
        return model, optimizer, int(config["gradient_accumulation_steps"]), float(config["warmup_fraction"]), float(config["gradient_clip_norm"]), None, True
    if bool(config.get("enable_optional_full_unfreeze", False)):
        raise ContractError("Optional full VideoMAE unfreeze remains disabled")
    model = VideoMAEClassifier(checkpoint=str(config["checkpoint"]), final_trainable_blocks=int(config["initial_trainable_blocks"]))
    optimizer = torch.optim.AdamW(model.optimizer_groups(backbone_lr=float(config["lr_backbone"]), head_lr=float(config["lr_head"])), weight_decay=float(config["weight_decay"]))
    return model, optimizer, int(config["gradient_accumulation_steps"]), float(config["warmup_fraction"]), float(config["gradient_clip_norm"]), None, True


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        policy = load_frozen_policy(args.policy)
        model_name = str(policy["selected_model"])
        config = load_yaml(MODEL_CONFIGS[model_name])
        seed = int(config.get("seed", 42))
        seed_everything(seed)
        sources = load_source_metadata(args.metadata, require_files=False)
        synthetic_sources = [source for source in sources if source.domain == "synthetic"]
        synthetic_ids = tuple(sorted(source.source_id for source in synthetic_sources))
        synthetic_families = tuple(
            sorted({str(source.family_id) for source in synthetic_sources})
        )
        external_ids = tuple(sorted(source.source_id for source in sources if source.domain == "real_staged"))
        if not synthetic_ids:
            raise ContractError("Final training requires at least one validated synthetic source")
        manifest = FoldManifest(
            0,
            synthetic_ids,
            (),
            (),
            external_ids,
            seed,
            train_families=synthetic_families,
        )
        dataset = WindowClipDataset(args.window_index, args.windows_root, manifest, "train", training=True, augmentation=load_yaml(args.data_config)["augmentation"])
        batch_size = int(config["stage_2"]["batch_size"] if model_name == "resnet50" else config.get("batch_size", config.get("physical_batch_size")))
        loader = DataLoader(dataset, batch_size=batch_size, sampler=source_class_balanced_sampler(dataset), num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
        complete_config = _complete_config(args, config)
        if canonical_sha256(complete_config) != policy["selected_config_sha256"]:
            raise ContractError("Current config bundle differs from the selected CV config hash; final training refused")
        policy_sha = canonical_sha256(policy)
        run_dir = args.output_root / "final_model" / model_name
        resume_path = run_dir / "last.pt" if args.resume else None
        if args.resume:
            if not resume_path.is_file():
                raise FileNotFoundError(f"--resume requested but missing {resume_path}")
            environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        else:
            environment = initialize_run_directory(run_dir, config=complete_config, training_manifest={"schema_version": 2, "purpose": "final_all_synthetic_no_validation", "seed": seed, "train": list(synthetic_ids), "train_families": list(synthetic_families), "validation": [], "validation_families": [], "test": [], "test_families": [], "external_real_excluded": list(external_ids)}, seed=seed)
            (run_dir / "frozen_policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        attach_run_log(run_dir / "run.log")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model, optimizer, accumulation, warmup, clip_norm, callback, linear = _model_optimizer(model_name, config)
        model.to(device)
        result = train_fixed_policy(model, loader, optimizer=optimizer, model_name=model_name, run_dir=run_dir, device=device, epochs=int(policy["median_cv_best_epoch"]), frozen_threshold=float(policy["median_fold_validation_threshold"]), config_sha256=str(environment["config_sha256"]), policy_sha256=policy_sha, gradient_accumulation_steps=accumulation, warmup_fraction=warmup, gradient_clip_norm=clip_norm, resume_path=resume_path, epoch_start_callback=callback, use_linear_schedule=linear)
        result["training_family_count"] = len(synthetic_families)
        (run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checkpoint = run_dir / "final.pt"
        provenance = {"model_name": model_name, "checkpoint_path": checkpoint.name, "config_snapshot_path": "config_snapshot.json", "median_cv_best_epoch": int(policy["median_cv_best_epoch"]), "median_fold_validation_threshold": float(policy["median_fold_validation_threshold"]), "cv_summary_sha256": str(policy["cv_summary_sha256"]), "final_policy_sha256": policy_sha, "config_sha256": str(environment["config_sha256"]), "final_training_uses_validation_set": False, "training_source_count": result["training_source_count"], "training_family_count": result["training_family_count"]}
        (run_dir / "final_model_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logging.info(
            "Completed final model=%s epochs=%d frozen_threshold=%.6f sources=%d families=%d",
            model_name,
            int(policy["median_cv_best_epoch"]),
            float(policy["median_fold_validation_threshold"]),
            int(result["training_source_count"]),
            int(result["training_family_count"]),
        )
    except (ContractError, FileNotFoundError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        logging.error("Final training refused or failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
