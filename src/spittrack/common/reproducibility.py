"""Seed control and self-describing run provenance."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from spittrack.common.hashing import canonical_sha256


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG streams needed for deterministic epoch-boundary resume."""
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a checkpointed RNG bundle before the next DataLoader iterator is created."""
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def capture_run_environment() -> dict[str, Any]:
    """Capture versions and hardware without requiring a Git repository or CUDA."""
    import torch
    import torchvision

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
    }


def initialize_run_directory(
    run_dir: Path,
    *,
    config: dict[str, Any],
    fold_manifest_path: Path | None = None,
    training_manifest: dict[str, Any] | None = None,
    seed: int,
) -> dict[str, Any]:
    """Create a self-describing run directory and return its provenance record."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = json.dumps(config, indent=2, sort_keys=True) + "\n"
    (run_dir / "config_snapshot.json").write_text(config_payload, encoding="utf-8")
    if (fold_manifest_path is None) == (training_manifest is None):
        raise ValueError("Provide exactly one of fold_manifest_path or training_manifest")
    if fold_manifest_path is not None:
        manifest_text = fold_manifest_path.read_text(encoding="utf-8")
        (run_dir / "fold_manifest.json").write_text(manifest_text, encoding="utf-8")
    else:
        (run_dir / "training_manifest.json").write_text(
            json.dumps(training_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    environment = capture_run_environment()
    provenance = {
        **environment,
        "seed": seed,
        "config_sha256": canonical_sha256(config),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze = _command_output([sys.executable, "-m", "pip", "freeze"])
    (run_dir / "pip_freeze.txt").write_text((freeze or "unavailable") + "\n", encoding="utf-8")
    return provenance
