from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .hub import resolve_data_path
from .object_labels import parse_object_labels
from .transforms import build_transform


_DEFAULT_SPLIT_FILES = {"train", "val", "test"}


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.ipc as ipc  # noqa: F401
    except Exception as e:  # pragma: no cover - exercised only without optional dep
        raise ImportError("Stitched Arrow backend requires pyarrow. Install CAIDBench with `pip install -e .`.") from e


def _arrow_scalar_to_py(value: Any) -> Any:
    try:
        return value.as_py()
    except AttributeError:
        return value


def _column_pylist(batch: Any, name: str | None, default: Any, n: int) -> list[Any]:
    if name is None or name not in batch.schema.names:
        return [default] * n
    return batch.column(name).to_pylist()


def _normalize_task_hint(value: str, mode: str) -> str:
    if mode == "lower":
        return value.lower()
    if mode == "slug":
        return value.lower().replace(" ", "_").replace("-", "_")
    return value


@dataclass(frozen=True)
class StitchedArrowFile:
    path: Path
    dir_name: str
    split_name: str


@dataclass
class LoadedStitchedArrow:
    files: list[StitchedArrowFile]
    metadata: pd.DataFrame
    image_column: str
    label_column: str
    generator_column: str
    dataset_column: str
    source_path_column: str
    split_column: str


class StitchedArrowImageDataset(Dataset):
    """Map-style dataset for generator/split directories of Arrow IPC files.

    The metadata frame stores file and record-batch positions. Image bytes are
    loaded lazily from the matching Arrow file when a sample is requested.
    """

    def __init__(
        self,
        loaded: LoadedStitchedArrow,
        indices: Iterable[int],
        transform_cfg: dict[str, Any] | None = None,
        task_id: int | None = None,
        task_name: str | None = None,
    ) -> None:
        self.loaded = loaded
        self.indices = [int(i) for i in indices]
        self.transform = build_transform(transform_cfg)
        self.task_id = task_id
        self.task_name = task_name
        self._reader_cache: dict[int, tuple[Any, Any]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _reader(self, file_id: int) -> Any:
        cached = self._reader_cache.get(file_id)
        if cached is not None:
            return cached[1]
        _require_pyarrow()
        import pyarrow as pa
        import pyarrow.ipc as ipc

        source = pa.memory_map(str(self.loaded.files[file_id].path), "r")
        try:
            reader = ipc.open_file(source)
        except pa.ArrowInvalid as e:
            source.close()
            raise ValueError(
                "stitched_arrow requires Arrow IPC file format for random access. "
                f"Stream-format file is unsupported: {self.loaded.files[file_id].path}"
            ) from e
        self._reader_cache[file_id] = (source, reader)
        return reader

    def _batch_value(self, batch: Any, column: str, row: int, default: Any = None) -> Any:
        if column not in batch.schema.names:
            return default
        return _arrow_scalar_to_py(batch.column(column)[row])

    def _load_row(self, file_id: int, batch_index: int, batch_row: int) -> tuple[torch.Tensor, dict[str, Any]]:
        reader = self._reader(file_id)
        batch = reader.get_batch(batch_index)
        value = self._batch_value(batch, self.loaded.image_column, batch_row)
        if isinstance(value, dict):
            value = value.get("bytes")
        if isinstance(value, str):
            with Image.open(value) as img:
                x = self.transform(img)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(bytes(value))) as img:
                x = self.transform(img)
        else:
            raise ValueError(f"Unsupported image payload type in stitched_arrow: {type(value)!r}")
        actual = {
            "label": self._batch_value(batch, self.loaded.label_column, batch_row, -1),
            "generator": self._batch_value(batch, self.loaded.generator_column, batch_row, None),
            "dataset": self._batch_value(batch, self.loaded.dataset_column, batch_row, None),
            "path": self._batch_value(batch, self.loaded.source_path_column, batch_row, None),
            "split": self._batch_value(batch, self.loaded.split_column, batch_row, None),
        }
        return x, actual

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta_pos = self.indices[idx]
        meta = self.loaded.metadata.iloc[meta_pos].to_dict()
        file_id = int(meta["_file_id"])
        batch_index = int(meta["_batch_index"])
        batch_row = int(meta["_batch_row"])
        tid = int(self.task_id if self.task_id is not None else meta.get("task_id", -1))
        x, actual = self._load_row(file_id, batch_index, batch_row)
        label = int(actual["label"])
        generator = str(actual["generator"] or meta.get("generator", "unknown"))
        dataset = str(actual["dataset"] or meta.get("dataset", "unknown"))
        source_path = str(actual["path"] or meta.get("path", meta_pos))
        split = str(actual["split"] or meta.get("split", "unknown"))
        sample = {
            "x": x,
            "y": torch.tensor(label, dtype=torch.long),
            "task_id": torch.tensor(tid, dtype=torch.long),
            "domain": str(meta.get("domain", dataset)),
            "generator": generator,
            "scene": str(meta.get("scene", "unknown")),
            "path": source_path,
            "dataset": dataset,
            "source_dataset": dataset,
            "generator_name": generator,
            "source_path": source_path,
            "split": split,
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


class StitchedArrowDataSource:
    """Data source for directories shaped as `<generator>/<split>.arrow`.

    Expected Arrow columns default to the user's stitched dataset contract:
    `image`, `label`, `generator_name`, `source_dataset`, `source_path`, `split`.
    """

    def __init__(self, loaded: LoadedStitchedArrow) -> None:
        self.loaded = loaded
        self.metadata = loaded.metadata

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "StitchedArrowDataSource":
        cfg = dict(cfg)
        path = resolve_data_path(cfg)
        if path is None:
            raise ValueError("stitched_arrow data source requires data.path or data.remote")
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"stitched_arrow expects a directory root: {root}")

        image_column = str(cfg.get("image_column", "image"))
        label_column = str(cfg.get("label_column", "label"))
        generator_column = str(cfg.get("generator_column", "generator_name"))
        dataset_column = str(cfg.get("dataset_column", "source_dataset"))
        source_path_column = str(cfg.get("source_path_column", cfg.get("path_column", "source_path")))
        split_column = str(cfg.get("split_column", "split"))
        task_hint_mode = str(cfg.get("task_hint_mode", "dir"))
        domain_from = str(cfg.get("domain_from", "dir_name"))
        recursive = bool(cfg.get("recursive", False))
        require_splits = [str(x) for x in cfg.get("require_splits", [])]

        files = cls._discover_files(root, recursive=recursive, require_splits=require_splits)
        if not files:
            raise FileNotFoundError(f"No stitched split Arrow files found under {root}")
        loaded = LoadedStitchedArrow(
            files=files,
            metadata=cls._scan_metadata(
                files,
                image_column=image_column,
                label_column=label_column,
                generator_column=generator_column,
                dataset_column=dataset_column,
                source_path_column=source_path_column,
                split_column=split_column,
                task_hint_mode=task_hint_mode,
                domain_from=domain_from,
            ),
            image_column=image_column,
            label_column=label_column,
            generator_column=generator_column,
            dataset_column=dataset_column,
            source_path_column=source_path_column,
            split_column=split_column,
        )
        return cls(loaded)

    @staticmethod
    def _discover_files(root: Path, recursive: bool = False, require_splits: list[str] | None = None) -> list[StitchedArrowFile]:
        pattern = "**/*.arrow" if recursive else "*/*.arrow"
        grouped: dict[str, dict[str, Path]] = {}
        for path in sorted(root.glob(pattern)):
            if path.stem not in _DEFAULT_SPLIT_FILES:
                continue
            grouped.setdefault(path.parent.name, {})[path.stem] = path
        required = set(require_splits or [])
        out: list[StitchedArrowFile] = []
        for dir_name, split_paths in sorted(grouped.items()):
            if required and not required.issubset(split_paths):
                continue
            for split_name, path in sorted(split_paths.items()):
                out.append(StitchedArrowFile(path=path, dir_name=dir_name, split_name=split_name))
        return out

    @staticmethod
    def _scan_metadata(
        files: list[StitchedArrowFile],
        *,
        image_column: str,
        label_column: str,
        generator_column: str,
        dataset_column: str,
        source_path_column: str,
        split_column: str,
        task_hint_mode: str,
        domain_from: str,
    ) -> pd.DataFrame:
        _require_pyarrow()
        import pyarrow as pa
        import pyarrow.ipc as ipc

        rows: list[dict[str, Any]] = []
        for file_id, spec in enumerate(files):
            with pa.memory_map(str(spec.path), "r") as source:
                try:
                    reader = ipc.open_file(source)
                except pa.ArrowInvalid as e:
                    raise ValueError(
                        "stitched_arrow requires Arrow IPC file format for random access. "
                        f"Stream-format file is unsupported: {spec.path}"
                    ) from e
                if image_column not in reader.schema.names:
                    raise ValueError(f"Missing image column {image_column!r} in {spec.path}")
                if label_column not in reader.schema.names:
                    raise ValueError(f"Missing label column {label_column!r} in {spec.path}")
                local_rowid = 0
                for batch_index in range(reader.num_record_batches):
                    batch = reader.get_batch(batch_index)
                    n = batch.num_rows
                    for batch_row in range(n):
                        dir_name = spec.dir_name
                        generator = dir_name
                        dataset = "unknown"
                        split = spec.split_name
                        domain = dir_name if domain_from in {"dir_name", "generator"} else dataset
                        task_hint = _normalize_task_hint(dir_name, task_hint_mode)
                        rows.append(
                            {
                                "_file_id": file_id,
                                "_batch_index": batch_index,
                                "_batch_row": batch_row,
                                "_rowid": local_rowid,
                                "label": -1,
                                "split": split,
                                "dataset": dataset,
                                "source_dataset": dataset,
                                "domain": domain,
                                "generator": generator,
                                "generator_name": generator,
                                "dir_name": dir_name,
                                "task_hint": task_hint,
                                "path": "",
                                "source_path": "",
                                "arrow_file": str(spec.path),
                                "task_id": -1,
                            }
                        )
                        local_rowid += 1
        return pd.DataFrame(rows).reset_index(drop=True)

    def make_dataset(
        self,
        row_indices: Iterable[int],
        transform_cfg: dict[str, Any] | None = None,
        task_id: int | None = None,
        task_name: str | None = None,
    ) -> StitchedArrowImageDataset:
        return StitchedArrowImageDataset(self.loaded, indices=row_indices, transform_cfg=transform_cfg, task_id=task_id, task_name=task_name)
