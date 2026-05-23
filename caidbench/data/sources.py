from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
from torch.utils.data import Dataset


class DataSource:
    """Abstract Arrow metadata-backed data source used by protocol YAMLs."""

    metadata: pd.DataFrame

    def make_dataset(self, row_indices: Iterable[int], transform_cfg: dict[str, Any] | None = None, task_id: int | None = None, task_name: str | None = None) -> Dataset:
        raise NotImplementedError


def build_data_source(cfg: dict[str, Any]) -> DataSource:
    backend = str(cfg.get("backend", cfg.get("type", "aid_arrow"))).lower()
    if backend in {"arrow", "hf_arrow", "huggingface", "aid", "aid_arrow", "aid_dataset"}:
        from .arrow import ArrowDataSource

        return ArrowDataSource.from_config(cfg)
    raise ValueError(f"Unsupported data backend: {backend}")
