"""Center-frame ImageNet ResNet50 research baseline."""

from __future__ import annotations

import torch
from torch import nn


class ResNet50Classifier(nn.Module):
    """ResNet50 with the protocol's Dropout(0.3) + Linear(2048, 2) head."""

    def __init__(self, *, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torchvision is required for ResNet50") from exc
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 2))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Classify the center frame of `[B,T,C,H,W]` or an image batch `[B,C,H,W]`."""
        if clips.ndim == 5:
            clips = clips[:, clips.shape[1] // 2]
        if clips.ndim != 4:
            raise ValueError(f"Expected 4D images or 5D clips, got shape {tuple(clips.shape)}")
        return self.backbone(clips)

    def freeze_backbone(self) -> None:
        """Freeze all ResNet parameters except the replacement classification head."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def unfreeze_layer4_and_head(self) -> None:
        """Match stage two: keep early layers frozen and train layer4 plus the head."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for module in (self.backbone.layer4, self.backbone.fc):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def optimizer_groups(self, *, backbone_lr: float, head_lr: float) -> list[dict[str, object]]:
        """Return non-overlapping stage-two parameter groups."""
        return [
            {"params": list(self.backbone.layer4.parameters()), "lr": backbone_lr},
            {"params": list(self.backbone.fc.parameters()), "lr": head_lr},
        ]

