from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from spittrack.training.temporal_trainer import train_temporal_model


class TinyClipDataset(Dataset):
    def __init__(self, count: int = 4) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        label = index % 2
        clip = torch.full((16, 3, 4, 4), float(label))
        metadata = {
            "window_id": f"w{index}",
            "source_id": f"S{index % 2}",
            "track_id": index,
            "start_s": index * 0.5,
            "end_s": index * 0.5 + 2,
            "bbox_x1": 0.0,
            "bbox_y1": 0.0,
            "bbox_x2": 10.0,
            "bbox_y2": 20.0,
            "negative_subtype": "normal_activity",
        }
        return clip, torch.tensor(label), metadata


class TinyTemporalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 2)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        value = clips.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return self.classifier(value)


def test_temporal_trainer_writes_resumable_outputs_and_outer_predictions(tmp_path: Path) -> None:
    train_loader = DataLoader(TinyClipDataset(3), batch_size=1)
    evaluation_loader = DataLoader(TinyClipDataset(4), batch_size=2)
    model = TinyTemporalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    result = train_temporal_model(
        model,
        train_loader,
        evaluation_loader,
        evaluation_loader,
        model_name="tiny",
        optimizer=optimizer,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        max_epochs=2,
        patience=2,
        gradient_accumulation_steps=2,
        warmup_fraction=0.1,
        gradient_clip_norm=1.0,
        config_sha256="config",
        manifest_sha256="manifest",
        runtime_batch_size=1,
    )
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "outer_test_predictions.csv").is_file()
    assert result["model"] == "tiny"
    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["gradient_accumulation_steps"] == 2
