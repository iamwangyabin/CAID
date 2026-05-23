from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _scalar(value: Any) -> int | float | None:
    if torch.is_tensor(value):
        if value.ndim != 0:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class NullExperimentLogger:
    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


class SwanLabExperimentLogger:
    def __init__(self, cfg: Mapping[str, Any], output_dir: Path, method_name: str) -> None:
        logging_cfg = cfg.get("logging", {}) or {}
        if not isinstance(logging_cfg, Mapping):
            raise TypeError("logging config must be a mapping")
        raw = dict(logging_cfg)
        try:
            swanlab = importlib.import_module("swanlab")
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is enabled by default, but the 'swanlab' package is not installed. "
                "Install project dependencies or set logging.backend=none to disable experiment logging."
            ) from exc

        experiment_name = raw.get("experiment_name") or raw.get("name") or f"{method_name}-{output_dir.name}"
        kwargs: dict[str, Any] = {
            "project": raw.get("project", "CAIDBench"),
            "experiment_name": experiment_name,
            "description": raw.get("description"),
            "job_type": raw.get("job_type", "train"),
            "group": raw.get("group"),
            "tags": raw.get("tags"),
            "workspace": raw.get("workspace"),
            "logdir": str(raw.get("logdir") or output_dir / "swanlog"),
            "mode": raw.get("mode", "cloud"),
            "config": _jsonable(cfg),
            "reinit": raw.get("reinit", True),
        }
        for key in ("id", "resume", "public", "parallel", "color"):
            if key in raw:
                kwargs[key] = raw[key]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self._swanlab = swanlab
        self._run = swanlab.init(**kwargs)

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        payload: dict[str, int | float] = {}
        for key, value in data.items():
            scalar = _scalar(value)
            if scalar is not None:
                payload[str(key)] = scalar
        if payload:
            self._swanlab.log(payload, step=step)

    def finish(self) -> None:
        if self._run is not None and hasattr(self._run, "finish"):
            self._run.finish()
        else:
            self._swanlab.finish()


def build_experiment_logger(cfg: Mapping[str, Any], output_dir: Path, method_name: str):
    logging_cfg = cfg.get("logging", {})
    if logging_cfg is False:
        return NullExperimentLogger()
    raw = logging_cfg or {}
    if not isinstance(raw, Mapping):
        raise TypeError("logging config must be a mapping")
    backend = str(raw.get("backend", "swanlab")).lower()
    if backend in {"none", "null", "disabled", "off", "false"}:
        return NullExperimentLogger()
    if backend != "swanlab":
        raise ValueError(f"Unsupported logging backend: {backend}")
    return SwanLabExperimentLogger(cfg, output_dir, method_name)
