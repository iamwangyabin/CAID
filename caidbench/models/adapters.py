from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class AdapterBlock(nn.Module):
    """Residual bottleneck adapter used by content-agnostic and PEFT-style methods."""

    def __init__(self, dim: int, bottleneck: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.scale = float(scale)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up(F.gelu(self.down(x)))


class LoRALinear(nn.Module):
    """LoRA adapter around a linear layer.

    The base layer can be frozen externally.  The LoRA branch is initialized as
    near-zero, matching common LoRA fine-tuning behavior.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, bias: bool = True) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.scaling = float(alpha) / max(int(rank), 1)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(x))

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.lora_a.parameters()
        yield from self.lora_b.parameters()


class MultiSceneLoRAHead(nn.Module):
    """Scene-indexed LoRA heads over shared features."""

    def __init__(self, feature_dim: int, num_classes: int = 2, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.heads = nn.ModuleDict()
        self.add_scene("unknown")

    def add_scene(self, scene: str) -> None:
        if scene not in self.heads:
            self.heads[scene] = LoRALinear(self.feature_dim, self.num_classes, rank=self.rank, alpha=self.alpha)

    def forward(self, features: torch.Tensor, scenes: list[str] | tuple[str, ...] | None = None) -> torch.Tensor:
        if scenes is None:
            scenes = ["unknown"] * features.shape[0]
        logits = []
        for z, s in zip(features, scenes):
            s = str(s)
            if s not in self.heads:
                self.add_scene(s)
                self.heads[s].to(features.device)
            logits.append(self.heads[s](z.unsqueeze(0)))
        return torch.cat(logits, dim=0)


def grid_shuffle(x: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """Grid shuffle for image tensors [B,C,H,W], used as token-shuffle fallback."""
    if x.dim() != 4:
        return x
    b, c, h, w = x.shape
    gh = gw = int(grid)
    if h % gh != 0 or w % gw != 0:
        return x.flip(-1)  # deterministic fallback when grid cannot divide
    patches = x.reshape(b, c, gh, h // gh, gw, w // gw).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c, h // gh, w // gw)
    perm = torch.randperm(gh * gw, device=x.device)
    patches = patches[:, perm]
    return patches.reshape(b, gh, gw, c, h // gh, w // gw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
