from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbones import build_backbone


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Detector(nn.Module):
    """Canonical detector returning both logits and features."""

    def __init__(self, backbone: nn.Module, num_classes: int = 2, head: nn.Module | None = None, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(getattr(backbone, "out_dim"))
        self.head = head if head is not None else MLPHead(self.feature_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, return_features: bool = True) -> dict[str, torch.Tensor]:
        z = self.extract_features(x)
        logits = self.head(z)
        return {"logits": logits, "features": z} if return_features else {"logits": logits}


def build_detector(cfg: dict[str, Any] | None = None) -> Detector:
    cfg = cfg or {}
    backbone = build_backbone(cfg.get("backbone", {}))
    return Detector(
        backbone=backbone,
        num_classes=int(cfg.get("num_classes", 2)),
        hidden_dim=cfg.get("hidden_dim"),
        dropout=float(cfg.get("dropout", 0.0)),
    )
