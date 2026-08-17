from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from spittrack.models.resnet_lstm import ResNet50LSTM
from spittrack.models.videomae import VideoMAEClassifier


def test_resnet_lstm_accepts_exactly_sixteen_frames() -> None:
    model = ResNet50LSTM(pretrained=False, hidden_size=8, num_layers=2)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 16, 3, 64, 64))
    assert output.shape == (1, 2)
    assert any(parameter.requires_grad for parameter in model.encoder.layer4.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.layer3.parameters())


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(8)])


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()


class FakeVideoMAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.videomae = FakeBackbone()
        self.classifier = nn.Linear(4, 2)

    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(1, 2, 3, 4)).unsqueeze(1).repeat(1, 4)
        return SimpleNamespace(logits=self.classifier(pooled))


def test_videomae_only_unfreezes_classifier_and_final_blocks(monkeypatch) -> None:
    from transformers import VideoMAEForVideoClassification

    monkeypatch.setattr(
        VideoMAEForVideoClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeVideoMAE(),
    )
    model = VideoMAEClassifier(checkpoint="fake", final_trainable_blocks=4)
    blocks = model.encoder_blocks
    assert not any(parameter.requires_grad for block in blocks[:4] for parameter in block.parameters())
    assert all(parameter.requires_grad for block in blocks[-4:] for parameter in block.parameters())
    output = model(torch.zeros(2, 16, 3, 8, 8))
    assert output.shape == (2, 2)
