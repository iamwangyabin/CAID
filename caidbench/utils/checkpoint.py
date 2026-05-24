from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu", *, weights_only: bool | None = True) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:  # Older torch releases do not accept weights_only.
        obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint {path} did not contain a dict")
    return obj
