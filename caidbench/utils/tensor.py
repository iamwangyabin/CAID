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
