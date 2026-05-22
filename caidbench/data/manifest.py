from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .object_labels import parse_object_labels
from .transforms import build_transform

_REQUIRED = {"path", "label", "split"}


def read_manifest(path: str | Path, require_task_id: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
    missing = _REQUIRED - set(df.columns)
    if require_task_id and "task_id" not in df.columns:
        missing = set(missing) | {"task_id"}
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    df = df.copy()
    if "task_id" not in df.columns:
        df["task_id"] = -1
    df["task_id"] = df["task_id"].astype(int)
    df["label"] = df["label"].astype(int)
    for col in ["domain", "generator", "scene"]:
        if col not in df.columns:
            df[col] = "unknown"
    return df


class ManifestImageDataset(Dataset):
    """Dataset backed by a manifest.

    The `path` column may point to an image file or a `.npy` feature array.
    Returned samples use canonical keys: `x`, `y`, `task_id`, `domain`,
    `generator`, `scene`, `path`, and optional extra manifest columns.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str | None = None,
        task_id: int | None = None,
        transform_cfg: dict[str, Any] | None = None,
        root: str | Path | None = None,
        indices: Iterable[int] | None = None,
    ) -> None:
        if isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
        else:
            df = read_manifest(manifest)
        if split is not None:
            df = df[df["split"].astype(str) == str(split)]
        if task_id is not None:
            df = df[df["task_id"].astype(int) == int(task_id)]
        if indices is not None:
            df = df.iloc[list(indices)]
        self.df = df.reset_index(drop=True)
        self.root = Path(root) if root else None
        self.transform = build_transform(transform_cfg)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if self.root is not None and not path.is_absolute():
            path = self.root / path
        return path

    def _load_x(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
            return torch.as_tensor(arr, dtype=torch.float32)
        if suffix in {".pt", ".pth"}:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                for key in ("x", "feature", "features", "image"):
                    if key in obj:
                        obj = obj[key]
                        break
            return torch.as_tensor(obj, dtype=torch.float32)
        with Image.open(path) as img:
            return self.transform(img)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx].to_dict()
        path = self._resolve_path(str(row["path"]))
        sample = {
            "x": self._load_x(path),
            "y": torch.tensor(int(row["label"]), dtype=torch.long),
            "task_id": torch.tensor(int(row["task_id"]), dtype=torch.long),
            "domain": str(row.get("domain", row.get("generator", "unknown"))),
            "generator": str(row.get("generator", row.get("domain", "unknown"))),
            "scene": str(row.get("scene", "unknown")),
            "path": str(path),
        }
        object_labels = parse_object_labels(row)
        if object_labels is not None:
            sample["object_labels"] = object_labels
        for k, v in row.items():
            if k not in sample and k not in {"label"}:
                sample[k] = v
        return sample


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            try:
                out[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                out[k] = vals
        elif isinstance(vals[0], (int, float, np.number)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


def build_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0, drop_last: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=_collate)
