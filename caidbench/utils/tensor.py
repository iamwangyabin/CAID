from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_to_device(obj: Any, device: torch.device | str, *, non_blocking: bool = False) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device, non_blocking=non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device, non_blocking=non_blocking) for v in obj)
    return obj
