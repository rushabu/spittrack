"""Materialized-window dataset with clip-consistent augmentation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from spittrack.common.errors import ContractError
from spittrack.data.folds import FoldManifest


class WindowClipDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """Load materialized RGB track clips selected by immutable source manifests."""

    def __init__(
        self,
        index_path: Path,
        clips_root: Path,
        manifest: FoldManifest,
        partition: str,
        *,
        training: bool = False,
    ) -> None:
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown partition {partition!r}")
        manifest.validate()
        partition_ids = set(getattr(manifest, partition))
        assigned = set(manifest.train) | set(manifest.validation) | set(manifest.test)
        with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"window_id", "source_id", "track_id", "label", "clip_path"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ContractError(f"Window index is missing required columns {sorted(required)}")
            all_rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
        unknown = sorted({row["source_id"] for row in all_rows} - assigned)
        external = sorted({row["source_id"] for row in all_rows} & set(manifest.external_real))
        if unknown:
            raise ContractError(f"Window index contains sources unassigned by fold {manifest.fold}: {unknown}")
        if external:
            raise ContractError(f"Synthetic window index contains external-real sources: {external}")
        self.rows = [
            row
            for row in all_rows
            if row["source_id"] in partition_ids and row["label"] != "ignore"
        ]
        if not self.rows:
            raise ContractError(f"No usable windows for fold {manifest.fold} partition {partition}")
        invalid_labels = sorted({row["label"] for row in self.rows} - {"spitting", "non_spitting"})
        if invalid_labels:
            raise ContractError(f"Window index contains invalid training labels: {invalid_labels}")
        self.clips_root = clips_root
        self.training = training
        self.labels = [1 if row["label"] == "spitting" else 0 for row in self.rows]
        self.source_ids = [row["source_id"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def _transform(self, clip: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import functional as F
        from torchvision.transforms import RandomResizedCrop
        from torchvision.transforms.functional import InterpolationMode

        clip = clip.float().div_(255.0)
        if self.training:
            i, j, h, w = RandomResizedCrop.get_params(
                clip[0], scale=(0.85, 1.0), ratio=(0.9, 1.1)
            )
            clip = F.resized_crop(
                clip, i, j, h, w, [224, 224], interpolation=InterpolationMode.BILINEAR
            )
            if random.random() < 0.5:
                clip = F.hflip(clip)
            angle = random.uniform(-5.0, 5.0)
            clip = F.rotate(clip, angle, interpolation=InterpolationMode.BILINEAR)
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            clip = F.adjust_brightness(clip, brightness)
            clip = F.adjust_contrast(clip, contrast)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype).view(1, 3, 1, 1)
        return (clip - mean) / std

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows[index]
        path = self.clips_root / row["clip_path"]
        if not path.is_file():
            raise ContractError(f"Materialized clip is missing: {path}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (16, 224, 224, 3) or array.dtype != np.uint8:
            raise ContractError(
                f"Clip {path} must be uint8 [16,224,224,3], got {array.dtype} {array.shape}"
            )
        clip = torch.from_numpy(array.copy()).permute(0, 3, 1, 2)
        metadata = {
            "window_id": row["window_id"],
            "source_id": row["source_id"],
            "track_id": int(row["track_id"]),
        }
        return self._transform(clip), torch.tensor(self.labels[index], dtype=torch.long), metadata


def source_class_balanced_sampler(dataset: WindowClipDataset) -> WeightedRandomSampler:
    """Give each class equal mass, each source equal mass within class, then each window equal source mass."""
    source_class: dict[str, int] = {}
    windows_per_source: dict[str, int] = {}
    sources_per_class = {0: set(), 1: set()}
    for source_id, label in zip(dataset.source_ids, dataset.labels):
        if source_id in source_class and source_class[source_id] != label:
            raise ContractError(f"Source {source_id} contains conflicting window classes")
        source_class[source_id] = label
        windows_per_source[source_id] = windows_per_source.get(source_id, 0) + 1
        sources_per_class[label].add(source_id)
    if not sources_per_class[0] or not sources_per_class[1]:
        raise ContractError("Training partition must contain both classes")
    weights = [
        1.0 / (2 * len(sources_per_class[label]) * windows_per_source[source_id])
        for source_id, label in zip(dataset.source_ids, dataset.labels)
    ]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )

