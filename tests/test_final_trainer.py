from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from spittrack.training.final_trainer import train_fixed_policy


class FinalDataset(Dataset):
    source_ids = ["S1", "S2", "S1", "S2"]
    labels = [0, 1, 0, 1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.full((16, 3, 2, 2), float(self.labels[index])), torch.tensor(self.labels[index]), {}


class FinalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, 2)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.head(clips.mean(dim=(1, 2, 3, 4)).unsqueeze(1))


def test_fixed_final_training_has_no_best_or_validation_artifact(tmp_path: Path) -> None:
    model = FinalModel()
    result = train_fixed_policy(
        model,
        DataLoader(FinalDataset(), batch_size=2),
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
        model_name="tiny",
        run_dir=tmp_path,
        device=torch.device("cpu"),
        epochs=2,
        frozen_threshold=0.73,
        config_sha256="config",
        policy_sha256="policy",
    )
    assert (tmp_path / "final.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert not (tmp_path / "best.pt").exists()
    checkpoint = torch.load(tmp_path / "final.pt", map_location="cpu", weights_only=False)
    assert checkpoint["frozen_threshold"] == 0.73
    assert checkpoint["final_training_uses_validation_set"] is False
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "validation" not in metrics and "test" not in metrics
    assert result["training_source_count"] == 2
