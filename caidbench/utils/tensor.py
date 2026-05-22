from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_to_device(obj: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.long().view(-1)
    out = torch.zeros(labels.numel(), num_classes, device=labels.device, dtype=torch.float32)
    out.scatter_(1, labels[:, None], 1.0)
    return out


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)
