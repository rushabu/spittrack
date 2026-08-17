"""Shared ResNet50 frame encoder followed by the protocol's two-layer LSTM."""

from __future__ import annotations

import torch
from torch import nn


class ResNet50LSTM(nn.Module):
    """Encode 16 frames to 2048-D features and classify the final LSTM state."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torchvision is required for ResNet50-LSTM") from exc
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.encoder = resnet50(weights=weights)
        feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout), nn.Linear(hidden_size, 2)
        )
        self.freeze_early_backbone()

    def freeze_early_backbone(self) -> None:
        """Train layer4 plus LSTM/head while keeping earlier ImageNet features frozen."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder.layer4.parameters():
            parameter.requires_grad = True
        for module in (self.lstm, self.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or clips.shape[1] != 16:
            raise ValueError(f"Expected [B,16,C,H,W], got {tuple(clips.shape)}")
        batch, frames, channels, height, width = clips.shape
        features = self.encoder(clips.reshape(batch * frames, channels, height, width))
        sequence = features.reshape(batch, frames, -1)
        output, _state = self.lstm(sequence)
        return self.classifier(output[:, -1])

    def optimizer_groups(self, *, backbone_lr: float, head_lr: float) -> list[dict[str, object]]:
        return [
            {"params": list(self.encoder.layer4.parameters()), "lr": backbone_lr},
            {
                "params": list(self.lstm.parameters()) + list(self.classifier.parameters()),
                "lr": head_lr,
            },
        ]
