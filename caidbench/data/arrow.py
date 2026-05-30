from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .object_labels import parse_object_labels
from .transforms import build_transform
from .arrow_schema import AID_MAPPING_FILE, read_aid_split_sidecars, read_caid_meta_sidecar
from .hub import resolve_data_path

_IMAGE_LIKE_DEFAULTS = ("image", "image_bytes", "bytes", "jpg", "png")


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.ipc as ipc  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as e:  # pragma: no cover - exercised only without optional dep
        raise ImportError("Arrow backend requires pyarrow. Install CAIDBench project dependencies with `pip install -e .`.") from e


def _read_arrow_file(path: Path):
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pq.read_table(path)
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_stream(source).read_all()


def _read_arrow_path(path: str | Path):
    _require_pyarrow()
    import pyarrow as pa

    path = Path(path)
    if path.is_dir():
        # AID/HuggingFace dataset directory.  The Arrow payload usually has only
        # one column: `image`; metadata is read from index.jsonl separately.
        if (path / "dataset_info.json").exists() and (path / "state.json").exists():
            try:
                from datasets import load_from_disk

                ds = load_from_disk(str(path))
                return ds.data.table if hasattr(ds, "data") else ds._data.table
            except Exception:
                pass
        files = sorted([*path.glob("*.arrow"), *path.glob("*.feather"), *path.glob("*.parquet")])
        if not files:
            files = sorted([*path.rglob("*.arrow"), *path.rglob("*.parquet")])
        if not files:
            raise FileNotFoundError(f"No .arrow/.parquet files found under {path}")
        return pa.concat_tables([_read_arrow_file(f) for f in files], promote_options="default")
    return _read_arrow_file(path)


def _arrow_scalar_to_py(value: Any) -> Any:
    try:
        return value.as_py()
    except AttributeError:
        return value


def _safe_metadata(table, skip_columns: set[str], sidecar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return metadata rows for protocol filtering.

    In strict AID mode, metadata is reconstructed from mapping.json and
    <split>.json.  Optional CAID metadata can add fields such as dataset,
    generator, video_id, frame_idx, but it is not required for AID compatibility.
    """
    if sidecar is not None:
        df = sidecar.copy()
    else:
        keep = [c for c in table.column_names if c not in skip_columns]
        df = table.select(keep).to_pandas() if keep else pd.DataFrame(index=range(table.num_rows))
    if "label" not in df.columns:
        raise ValueError(
            "Arrow source has no label metadata. For strict AID datasets, ensure mapping.json and <split>.json exist next to the saved HF dataset."
        )
    if "split" not in df.columns:
        df["split"] = "train"
    if "task_id" not in df.columns:
        df["task_id"] = -1
    for col in ["path", "dataset", "domain", "generator", "scene", "manipulation", "video_id", "frame_idx", "preprocess_profile", "task_hint"]:
        if col not in df.columns:
            df[col] = "unknown" if col not in {"frame_idx", "task_hint", "path", "video_id", "preprocess_profile"} else (-1 if col == "frame_idx" else "")
    if "domain" in df.columns and "dataset" in df.columns:
        df["domain"] = df["domain"].where(df["domain"].astype(str) != "unknown", df["dataset"].astype(str))
    df["label"] = df["label"].astype(int)
    df["task_id"] = df["task_id"].fillna(-1).astype(int)
    if "_rowid" not in df.columns:
        if len(df) != table.num_rows:
            raise ValueError(f"Metadata rows ({len(df)}) do not match Arrow rows ({table.num_rows}); no _rowid column was provided")
        df["_rowid"] = range(len(df))
    df["_rowid"] = df["_rowid"].astype(int)
    return df.reset_index(drop=True)


def _enrich_aid_sidecar(sidecar: pd.DataFrame, rich: pd.DataFrame | None) -> pd.DataFrame:
    if rich is None or "path" not in rich.columns:
        return sidecar
    merge_keys = ["path"]
    if "split" in sidecar.columns and "split" in rich.columns:
        merge_keys.append("split")
    rich_cols = [c for c in rich.columns if c in merge_keys or c != "_rowid"]
    merged = sidecar.merge(rich[rich_cols], on=merge_keys, how="left", suffixes=("", "_rich"))
    for col in [c for c in rich_cols if c not in merge_keys]:
        rich_col = f"{col}_rich"
        if rich_col not in merged.columns:
            continue
        if col == "label" and "label" in merged.columns:
            base_label = pd.to_numeric(merged["label"], errors="coerce")
            rich_label = pd.to_numeric(merged[rich_col], errors="coerce")
            mask = base_label.eq(-1) & rich_label.notna()
            merged.loc[mask, "label"] = rich_label[mask].astype(int)
        elif col in merged.columns:
            merged[col] = merged[rich_col].where(merged[rich_col].notna(), merged[col])
        else:
            merged[col] = merged[rich_col]
        merged = merged.drop(columns=[rich_col])
    return merged

@dataclass
class LoadedArrow:
    table: Any
    metadata: pd.DataFrame
    image_column: str | None
    path_column: str | None
    root: Path | None


class ArrowImageDataset(Dataset):
    """PyTorch Dataset backed by Arrow/HuggingFace-style Arrow tables.

    Supported sample storage forms:
      - image bytes in image_bytes / image / bytes
      - HuggingFace Image-like dict {"bytes": ..., "path": ...}
      - image paths in path / image_path / file_name
    """

    def __init__(self, loaded: LoadedArrow, indices: Iterable[int], transform_cfg: dict[str, Any] | None = None, task_id: int | None = None, task_name: str | None = None) -> None:
        self.loaded = loaded
        self.indices = [int(i) for i in indices]
        self.transform = build_transform(transform_cfg)
        self.task_id = task_id
        self.task_name = task_name

    def __len__(self) -> int:
        return len(self.indices)

    def _column_value(self, col: str, row: int) -> Any:
        return _arrow_scalar_to_py(self.loaded.table[col][row])

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if self.loaded.root is not None and not path.is_absolute():
            path = self.loaded.root / path
        return path

    def _load_image_from_bytes(self, b: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(b)) as img:
            return self.transform(img)

    def _load_path(self, p: str) -> torch.Tensor:
        path = self._resolve_path(p)
        with Image.open(path) as img:
            return self.transform(img)

    def _load_x(self, row: int) -> torch.Tensor:
        if self.loaded.image_column:
            value = self._column_value(self.loaded.image_column, row)
            if isinstance(value, dict):
                if value.get("bytes") is not None:
                    return self._load_image_from_bytes(value["bytes"])
                if value.get("path"):
                    return self._load_path(value["path"])
            if isinstance(value, (bytes, bytearray, memoryview)):
                return self._load_image_from_bytes(bytes(value))
            if isinstance(value, str):
                return self._load_path(value)
        if self.loaded.path_column:
            return self._load_path(str(self._column_value(self.loaded.path_column, row)))
        raise ValueError("Arrow row has no image/path column configured")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta_pos = self.indices[idx]
        meta = self.loaded.metadata.iloc[meta_pos].to_dict()
        row = int(meta.get("_rowid", meta_pos))
        tid = int(self.task_id if self.task_id is not None else meta.get("task_id", -1))
        sample = {
            "x": self._load_x(row),
            "y": torch.tensor(int(meta["label"]), dtype=torch.long),
            "task_id": torch.tensor(tid, dtype=torch.long),
            "domain": str(meta.get("domain", meta.get("dataset", "unknown"))),
            "generator": str(meta.get("generator", "unknown")),
            "scene": str(meta.get("scene", "unknown")),
            "path": str(meta.get("path", meta.get("image_path", row))),
        }
        object_labels = parse_object_labels(meta)
        if object_labels is not None:
            sample["object_labels"] = object_labels
        if self.task_name is not None:
            sample["task_name"] = self.task_name
        for k, v in meta.items():
            if k not in sample and k != "label":
                sample[k] = v
        return sample


class ArrowDataSource:
    def __init__(self, loaded: LoadedArrow) -> None:
        self.loaded = loaded
        self.metadata = loaded.metadata

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ArrowDataSource":
        backend = str(cfg.get("backend", cfg.get("type", "aid_arrow"))).lower()
        require_aid_sidecars = backend in {"aid", "aid_arrow", "aid_dataset"}
        if cfg.get("feature_column") is not None:
            raise ValueError("Pre-extracted feature columns are no longer supported; provide raw images with image_column or path_column.")
        path = resolve_data_path(cfg)
        if path is None:
            raise ValueError("Arrow data source requires data.path or data.remote")
        path_obj = Path(path)
        if require_aid_sidecars and (not path_obj.is_dir() or not (path_obj / AID_MAPPING_FILE).exists()):
            raise FileNotFoundError(
                f"AID Arrow backend expects a dataset directory containing {AID_MAPPING_FILE} and split JSON sidecars: {path_obj}"
            )
        table = _read_arrow_path(path)
        sidecar = None
        if path_obj.is_dir() and (path_obj / AID_MAPPING_FILE).exists():
            # Strict AID compatibility: mapping.json + <split>.json are enough.
            sidecar = read_aid_split_sidecars(path_obj)
            # Optional CAID metadata, if present, enriches AID split metadata.
            sidecar = _enrich_aid_sidecar(sidecar, read_caid_meta_sidecar(path_obj))
        names = set(table.column_names)
        image_column = cfg.get("image_column")
        path_column = cfg.get("path_column")
        if image_column is None:
            image_column = next((c for c in _IMAGE_LIKE_DEFAULTS if c in names), None)
        if path_column is None:
            path_column = next((c for c in ("path", "image_path", "file_name", "filepath") if c in names), None)
        skip = {c for c in [image_column] if c}
        metadata = _safe_metadata(table, skip, sidecar=sidecar)
        loaded = LoadedArrow(
            table=table,
            metadata=metadata,
            image_column=image_column,
            path_column=path_column,
            root=Path(cfg["root_dir"]) if cfg.get("root_dir") else None,
        )
        return cls(loaded)

    def make_dataset(self, row_indices: Iterable[int], transform_cfg: dict[str, Any] | None = None, task_id: int | None = None, task_name: str | None = None) -> ArrowImageDataset:
        return ArrowImageDataset(self.loaded, indices=row_indices, transform_cfg=transform_cfg, task_id=task_id, task_name=task_name)
