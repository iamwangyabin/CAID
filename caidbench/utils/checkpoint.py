from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


_SAFE_SCALARS = (str, int, float, bool, type(None), bytes)


def _weights_only_value(value: Any, *, path: str) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_weights_only_value(item, path=f"{path}[{idx}]") for idx, item in enumerate(value.tolist())]
        return torch.as_tensor(value).cpu()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _SAFE_SCALARS):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _weights_only_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_weights_only_value(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
    raise TypeError(
        f"Checkpoint field {path!r} contains unsupported object {type(value).__name__}; "
        "checkpoints must contain only tensors, mappings, sequences, and scalar metadata."
    )


def weights_only_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _weights_only_value(value, path=str(key)) for key, value in payload.items()}


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights_only_payload(payload), path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu", *, weights_only: bool | None = True) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:  # Older torch releases do not accept weights_only.
        obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint {path} did not contain a dict")
    return obj
