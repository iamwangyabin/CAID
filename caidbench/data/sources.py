from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from torch.utils.data import Dataset

from .manifest import ManifestImageDataset, read_manifest


class DataSource:
    """Abstract metadata-backed data source used by protocol YAMLs."""

    metadata: pd.DataFrame

    def make_dataset(self, row_indices: Iterable[int], transform_cfg: dict[str, Any] | None = None, task_id: int | None = None, task_name: str | None = None) -> Dataset:
        raise NotImplementedError


@dataclass
class ManifestDataSource(DataSource):
    path: str | Path
    root: str | Path | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.metadata = read_manifest(self.path, require_task_id=False).reset_index(drop=True)
        self.metadata["_rowid"] = range(len(self.metadata))

    def make_dataset(self, row_indices: Iterable[int], transform_cfg: dict[str, Any] | None = None, task_id: int | None = None, task_name: str | None = None) -> Dataset:
        idx = list(int(i) for i in row_indices)
        df = self.metadata.iloc[idx].copy()
        if task_id is not None:
            df["task_id"] = int(task_id)
        if task_name is not None:
            df["task_name"] = str(task_name)
        return ManifestImageDataset(df, transform_cfg=transform_cfg, root=self.root)


def build_data_source(cfg: dict[str, Any]) -> DataSource:
    backend = str(cfg.get("backend", cfg.get("type", "manifest"))).lower()
    if backend in {"manifest", "csv", "jsonl"}:
        path = cfg.get("path", cfg.get("manifest"))
        if path is None:
            raise ValueError("Manifest data source requires data.path or data.manifest")
        return ManifestDataSource(path=path, root=cfg.get("root"))
    if backend in {"arrow", "hf_arrow", "huggingface", "aid", "aid_arrow", "aid_dataset"}:
        from .arrow import ArrowDataSource

        return ArrowDataSource.from_config(cfg)
    raise ValueError(f"Unsupported data backend: {backend}")
