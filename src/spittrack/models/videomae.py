"""Pretrained Hugging Face VideoMAE classification adapter."""

from __future__ import annotations

import torch
from torch import nn


class VideoMAEClassifier(nn.Module):
    """Two-class pretrained VideoMAE with protocol-controlled partial unfreezing."""

    def __init__(
        self,
        *,
        checkpoint: str = "MCG-NJU/videomae-base",
        final_trainable_blocks: int = 4,
    ) -> None:
        super().__init__()
        try:
            from transformers import VideoMAEForVideoClassification
        except ImportError as exc:  # pragma: no cover - dependency checked on Kaggle
            raise RuntimeError("transformers with VideoMAE support is required") from exc
        self.model = VideoMAEForVideoClassification.from_pretrained(
            checkpoint,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        self.set_partial_trainability(final_trainable_blocks)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Return logits for RGB clips shaped `[B,T,C,H,W]`."""
        if clips.ndim != 5 or clips.shape[1] != 16:
            raise ValueError(f"VideoMAE expects [B,16,C,H,W], got {tuple(clips.shape)}")
        return self.model(pixel_values=clips).logits

    @property
    def encoder_blocks(self) -> nn.ModuleList:
        return self.model.videomae.encoder.layer

    def set_partial_trainability(self, final_blocks: int = 4) -> None:
        """Train only classifier and final N transformer blocks initially."""
        if not 1 <= final_blocks <= len(self.encoder_blocks):
            raise ValueError(
                f"final_blocks must be between 1 and {len(self.encoder_blocks)}, got {final_blocks}"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True
        for block in self.encoder_blocks[-final_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def unfreeze_all(self) -> None:
        """Enable optional full fine-tuning after the declared stability gate."""
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def optimizer_groups(self, *, backbone_lr: float, head_lr: float) -> list[dict[str, object]]:
        """Separate classifier from all other currently trainable parameters."""
        head_ids = {id(parameter) for parameter in self.model.classifier.parameters()}
        backbone = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        head = [parameter for parameter in self.model.classifier.parameters() if parameter.requires_grad]
        return [
            {"params": backbone, "lr": backbone_lr},
            {"params": head, "lr": head_lr},
        ]

