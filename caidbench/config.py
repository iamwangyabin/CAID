from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must contain a mapping at top level")
    return data


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_update(dict(out[k]), v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    parent = cfg.pop("_base_", None)
    if parent:
        base_path = Path(path).parent / parent
        cfg = deep_update(load_config(base_path), cfg)
    return cfg


def add_common_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--override", nargs="*", default=[], help="Optional dotted key=value overrides")
    return parser


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    return cfg
