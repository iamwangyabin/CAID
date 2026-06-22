from __future__ import annotations

from contextlib import suppress
import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .hub import resolve_data_path
from .object_labels import parse_object_labels
from .transforms import build_transform


_DEFAULT_SPLIT_FILES = {"train", "val", "test"}
_DEFAULT_FACE_BBOX_FILENAME = "forgerynet_face_bboxes_all_generators.parquet"


class _CorruptImageError(RuntimeError):
    pass


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.ipc as ipc  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as e:  # pragma: no cover - exercised only without optional dep
        raise ImportError("CAIDBench Arrow backend requires pyarrow. Install CAIDBench with `pip install -e .`.") from e


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
class CAIDBenchArrowFile:
    path: Path
    dir_name: str
    split_name: str


@dataclass
class LoadedCAIDBenchArrow:
    files: list[CAIDBenchArrowFile]
    metadata: pd.DataFrame
    image_column: str
    label_column: str
    generator_column: str
    dataset_column: str
    source_path_column: str
    split_column: str
    prefer_metadata: bool = False
    face_bboxes: dict[tuple[str, int, int], tuple[float, float, float, float]] | None = None


class CAIDBenchArrowImageDataset(Dataset):
    """Map-style dataset for generator/split directories of Arrow IPC files.

    The metadata frame stores file and record-batch positions. Image bytes are
    loaded lazily from the matching Arrow file when a sample is requested.
    """

    def __init__(
        self,
        loaded: LoadedCAIDBenchArrow,
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

    def close(self) -> None:
        for source, reader in self._reader_cache.values():
            reader_close = getattr(reader, "close", None)
            if callable(reader_close):
                with suppress(Exception):
                    reader_close()
            source_close = getattr(source, "close", None)
            if callable(source_close):
                with suppress(Exception):
                    source_close()
        self._reader_cache.clear()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

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
                "CAIDBench Arrow requires Arrow IPC file format for random access. "
                f"Stream-format file is unsupported: {self.loaded.files[file_id].path}"
            ) from e
        self._reader_cache[file_id] = (source, reader)
        return reader

    def _batch_value(self, batch: Any, column: str, row: int, default: Any = None) -> Any:
        if column not in batch.schema.names:
            return default
        return _arrow_scalar_to_py(batch.column(column)[row])

    def _crop_face_if_available(
        self,
        img: Image.Image,
        *,
        batch_index: int,
        batch_row: int,
        meta: Mapping[str, Any],
    ) -> Image.Image:
        if not self.loaded.face_bboxes:
            return img
        arrow_path = meta.get("arrow_path")
        if arrow_path is None:
            return img
        bbox = self.loaded.face_bboxes.get((str(arrow_path), batch_index, batch_row))
        if bbox is None:
            return img
        x1, y1, x2, y2 = bbox
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            return img
        left = max(0, int(math.floor(x1)))
        top = max(0, int(math.floor(y1)))
        right = min(img.width, int(math.ceil(x2)))
        bottom = min(img.height, int(math.ceil(y2)))
        if right <= left or bottom <= top:
            return img
        return img.crop((left, top, right, bottom))

    def _load_row(
        self,
        file_id: int,
        batch_index: int,
        batch_row: int,
        meta: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        reader = self._reader(file_id)
        batch = reader.get_batch(batch_index)
        value = self._batch_value(batch, self.loaded.image_column, batch_row)
        if isinstance(value, dict):
            value = value.get("bytes")
        if isinstance(value, str):
            try:
                with Image.open(value) as img:
                    img = self._crop_face_if_available(
                        img,
                        batch_index=batch_index,
                        batch_row=batch_row,
                        meta=meta,
                    )
                    x = self.transform(img)
            except (OSError, UnidentifiedImageError) as e:
                raise _CorruptImageError(str(e)) from e
        elif isinstance(value, (bytes, bytearray, memoryview)):
            try:
                with Image.open(io.BytesIO(bytes(value))) as img:
                    img = self._crop_face_if_available(
                        img,
                        batch_index=batch_index,
                        batch_row=batch_row,
                        meta=meta,
                    )
                    x = self.transform(img)
            except (OSError, UnidentifiedImageError) as e:
                raise _CorruptImageError(str(e)) from e
        else:
            raise ValueError(f"Unsupported image payload type in CAIDBench Arrow: {type(value)!r}")
        actual = {
            "label": self._batch_value(batch, self.loaded.label_column, batch_row, -1),
            "generator": self._batch_value(batch, self.loaded.generator_column, batch_row, None),
            "dataset": self._batch_value(batch, self.loaded.dataset_column, batch_row, None),
            "path": self._batch_value(batch, self.loaded.source_path_column, batch_row, None),
            "split": self._batch_value(batch, self.loaded.split_column, batch_row, None),
        }
        return x, actual

    def _to_sample(self, idx: int, meta: Mapping[str, Any], x: torch.Tensor, actual: dict[str, Any], tid: int) -> dict[str, Any]:
        if self.loaded.prefer_metadata:
            label = int(meta.get("label", actual["label"]))
            generator = str(meta.get("generator", meta.get("generator_name", actual["generator"] or "unknown")))
            dataset = str(meta.get("dataset", actual["dataset"] or "unknown"))
            source_path = str(meta.get("path", meta.get("source_path", actual["path"] or idx)))
            split = str(meta.get("split", actual["split"] or "unknown"))
        else:
            actual_label = actual["label"]
            label = int(meta.get("label", -1) if actual_label in {None, -1} else actual_label)
            generator = str(actual["generator"] or meta.get("generator", "unknown"))
            dataset = str(actual["dataset"] or meta.get("dataset", "unknown"))
            source_path = str(actual["path"] or meta.get("path", idx))
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

    def _load_sample(self, meta_pos: int) -> dict[str, Any]:
        meta = self.loaded.metadata.iloc[meta_pos].to_dict()
        file_id = int(meta["_file_id"])
        batch_index = int(meta["_batch_index"])
        batch_row = int(meta["_batch_row"])
        try:
            x, actual = self._load_row(file_id, batch_index, batch_row, meta)
        except _CorruptImageError as e:
            raise _CorruptImageError(f"{self._describe_meta(meta_pos, meta)}: {e}") from e
        tid = int(self.task_id if self.task_id is not None else meta.get("task_id", -1))
        return self._to_sample(meta_pos, meta, x, actual, tid)

    def _describe_meta(self, meta_pos: int, meta: Mapping[str, Any] | None = None) -> str:
        if meta is None:
            try:
                meta = self.loaded.metadata.iloc[int(meta_pos)].to_dict()
            except Exception:
                meta = {}
        parts = [f"meta_pos={meta_pos}"]
        for key in ("source_path", "path", "arrow_file", "generator_name", "split"):
            value = meta.get(key)
            if value is None:
                continue
            text = str(value)
            if text:
                parts.append(f"{key}={text}")
        if "_batch_index" in meta and "_batch_row" in meta:
            parts.append(f"batch={meta['_batch_index']} row={meta['_batch_row']}")
        return " ".join(parts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_len = len(self.indices)
        if dataset_len == 0:
            raise IndexError("Cannot fetch item from an empty CAIDBenchArrowImageDataset")
        if idx < 0:
            idx += dataset_len
        if idx < 0 or idx >= dataset_len:
            raise IndexError(idx)

        last_error: _CorruptImageError | None = None
        for offset in range(dataset_len):
            sample_idx = (idx + offset) % dataset_len
            meta_pos = self.indices[sample_idx]
            try:
                return self._load_sample(meta_pos)
            except _CorruptImageError as e:
                last_error = e
                location = self._describe_meta(meta_pos)
                warnings.warn(
                    f"Skipping corrupted sample idx={sample_idx} ({location}): "
                    f"{e.__class__.__name__}: {e}"
                )
                continue

        raise RuntimeError(
            f"No valid sample found after skipping {dataset_len} corrupt candidate(s) from idx={idx}"
        ) from last_error


class CAIDBenchArrowDataSource:
    """Data source for CAIDBench indexed Arrow data.

    Expected Arrow columns default to the CAIDBench dataset contract:
    `image`, `label`, `generator_name`, `source_dataset`, `source_path`, `split`.
    """

    def __init__(self, loaded: LoadedCAIDBenchArrow) -> None:
        self.loaded = loaded
        self.metadata = loaded.metadata
        self._split_task_indices: dict[tuple[str, str], list[int]] = {
            (str(task_hint), str(split)): [int(i) for i in group.index.tolist()]
            for (task_hint, split), group in self.metadata.groupby(["task_hint", "split"], sort=False)
        }
        self._selector_split_indices: dict[str, dict[tuple[str, str], list[int]]] = {"task_hint": self._split_task_indices}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CAIDBenchArrowDataSource":
        cfg = dict(cfg)
        path = resolve_data_path(cfg)
        if path is None:
            raise ValueError("CAIDBench Arrow data source requires data.path or data.remote")
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"CAIDBench Arrow expects a directory root: {root}")

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
        index_path = cfg.get("index_path", cfg.get("index"))
        face_bboxes = cls._load_face_bboxes(root / _DEFAULT_FACE_BBOX_FILENAME)

        if index_path is not None:
            loaded = cls._load_index_metadata(
                root,
                Path(index_path),
                image_column=image_column,
                label_column=label_column,
                generator_column=generator_column,
                dataset_column=dataset_column,
                source_path_column=source_path_column,
                split_column=split_column,
                task_hint_mode=task_hint_mode,
                domain_from=domain_from,
                face_bboxes=face_bboxes,
            )
            return cls(loaded)

        files = cls._discover_files(root, recursive=recursive, require_splits=require_splits)
        if not files:
            raise FileNotFoundError(f"No CAIDBench split Arrow files found under {root}")
        loaded = LoadedCAIDBenchArrow(
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
            face_bboxes=face_bboxes,
        )
        return cls(loaded)

    @staticmethod
    def _load_face_bboxes(path: Path) -> dict[tuple[str, int, int], tuple[float, float, float, float]] | None:
        if not path.is_file():
            return None
        frame = pd.read_parquet(path)
        required = {"arrow_path", "batch_id", "row_in_batch", "face_found", "x1", "y1", "x2", "y2"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"CAIDBench face bbox parquet is missing required columns: {missing}")
        frame = frame[frame["face_found"].astype(bool)].copy()
        if frame.empty:
            return {}
        frame = frame.dropna(subset=["arrow_path", "batch_id", "row_in_batch", "x1", "y1", "x2", "y2"])
        bboxes: dict[tuple[str, int, int], tuple[float, float, float, float]] = {}
        for row in frame.itertuples(index=False):
            x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                continue
            bboxes[(str(row.arrow_path), int(row.batch_id), int(row.row_in_batch))] = (x1, y1, x2, y2)
        return bboxes

    @staticmethod
    def _discover_files(root: Path, recursive: bool = False, require_splits: list[str] | None = None) -> list[CAIDBenchArrowFile]:
        pattern = "**/*.arrow" if recursive else "*/*.arrow"
        grouped: dict[str, dict[str, Path]] = {}
        for path in sorted(root.glob(pattern)):
            if path.stem not in _DEFAULT_SPLIT_FILES:
                continue
            grouped.setdefault(path.parent.name, {})[path.stem] = path
        required = set(require_splits or [])
        out: list[CAIDBenchArrowFile] = []
        for dir_name, split_paths in sorted(grouped.items()):
            if required and not required.issubset(split_paths):
                continue
            for split_name, path in sorted(split_paths.items()):
                out.append(CAIDBenchArrowFile(path=path, dir_name=dir_name, split_name=split_name))
        return out

    @staticmethod
    def _scan_metadata(
        files: list[CAIDBenchArrowFile],
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
                        "CAIDBench Arrow requires Arrow IPC file format for random access. "
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

    @staticmethod
    def _resolve_index_path(root: Path, index_path: Path) -> Path:
        if index_path.is_absolute():
            return index_path
        if index_path.exists():
            return index_path
        return root / index_path

    @classmethod
    def _load_index_metadata(
        cls,
        root: Path,
        index_path: Path,
        *,
        image_column: str,
        label_column: str,
        generator_column: str,
        dataset_column: str,
        source_path_column: str,
        split_column: str,
        task_hint_mode: str,
        domain_from: str,
        face_bboxes: dict[tuple[str, int, int], tuple[float, float, float, float]] | None,
    ) -> LoadedCAIDBenchArrow:
        _require_pyarrow()
        import pyarrow.parquet as pq

        index_path = cls._resolve_index_path(root, index_path)
        if not index_path.is_file():
            raise FileNotFoundError(f"CAIDBench index_path does not exist: {index_path}")

        metadata = pq.read_table(index_path).to_pandas()
        required = {"arrow_path", "batch_id", "row_in_batch", "label", "split", "generator_name"}
        missing = sorted(required - set(metadata.columns))
        if missing:
            raise ValueError(f"CAIDBench index_path is missing required columns: {missing}")

        metadata = metadata.copy().reset_index(drop=True)
        metadata["arrow_path"] = metadata["arrow_path"].astype(str)
        files: list[CAIDBenchArrowFile] = []
        file_ids: dict[str, int] = {}
        for rel_path in dict.fromkeys(metadata["arrow_path"].tolist()):
            path = Path(rel_path)
            full_path = path if path.is_absolute() else root / path
            if not full_path.is_file():
                raise FileNotFoundError(f"Indexed Arrow file does not exist: {full_path}")
            file_ids[rel_path] = len(files)
            files.append(CAIDBenchArrowFile(path=full_path, dir_name=full_path.parent.name, split_name=full_path.stem))

        metadata["_file_id"] = metadata["arrow_path"].map(file_ids).astype("int32")
        metadata["_batch_index"] = metadata["batch_id"].astype("int32")
        metadata["_batch_row"] = metadata["row_in_batch"].astype("int32")
        metadata["_rowid"] = range(len(metadata))
        metadata["label"] = metadata["label"].astype("int64")
        metadata["split"] = metadata["split"].astype(str)
        metadata["generator_name"] = metadata["generator_name"].astype(str)
        if "generator" not in metadata.columns:
            metadata["generator"] = metadata["generator_name"]
        if "domain" not in metadata.columns:
            metadata["domain"] = metadata["generator_name"] if domain_from in {"dir_name", "generator", "generator_name"} else "unknown"
        if "dataset" not in metadata.columns:
            metadata["dataset"] = metadata.get("source_dataset", "unknown")
        if "source_dataset" not in metadata.columns:
            metadata["source_dataset"] = metadata["dataset"]
        if "path" not in metadata.columns:
            metadata["path"] = metadata["arrow_path"].astype(str) + "#" + metadata["batch_id"].astype(str) + ":" + metadata["row_in_batch"].astype(str)
        if "source_path" not in metadata.columns:
            metadata["source_path"] = metadata["path"]
        if "dir_name" not in metadata.columns:
            metadata["dir_name"] = metadata["generator_name"]
        if "task_hint" not in metadata.columns:
            metadata["task_hint"] = metadata["generator_name"].map(lambda x: _normalize_task_hint(str(x), task_hint_mode))
        if "scene" not in metadata.columns:
            metadata["scene"] = "unknown"
        if "task_id" not in metadata.columns:
            task_ids = {name: i for i, name in enumerate(dict.fromkeys(metadata["generator_name"].tolist()))}
            metadata["task_id"] = metadata["generator_name"].map(task_ids)
        metadata["task_id"] = metadata["task_id"].astype("int64")

        return LoadedCAIDBenchArrow(
            files=files,
            metadata=metadata,
            image_column=image_column,
            label_column=label_column,
            generator_column=generator_column,
            dataset_column=dataset_column,
            source_path_column=source_path_column,
            split_column=split_column,
            prefer_metadata=True,
            face_bboxes=face_bboxes,
        )

    def make_dataset(
        self,
        row_indices: Iterable[int],
        transform_cfg: dict[str, Any] | None = None,
        task_id: int | None = None,
        task_name: str | None = None,
    ) -> CAIDBenchArrowImageDataset:
        return CAIDBenchArrowImageDataset(
            self.loaded,
            indices=row_indices,
            transform_cfg=transform_cfg,
            task_id=task_id,
            task_name=task_name,
        )

    def select_indices(self, spec: Mapping[str, Any] | None) -> list[int] | None:
        """Fast path for simple CAIDBench generator/split protocol filters.

        The CAIDBench dataset is physically partitioned as `<generator>/<split>.arrow`.
        For the common protocol shape, avoid scanning the full metadata frame for
        every task and split; return the precomputed row-index list directly.
        """
        if spec is None:
            return None
        spec = dict(spec)
        if any(k in spec for k in ("exclude", "query", "where", "sample", "limit")):
            return None
        split = spec.get("split")
        if split is None:
            return None
        include = dict(spec.get("include", {}) or {})
        for k, v in spec.items():
            if k not in {"include", "split"}:
                include[k] = v
        selector = None
        values = None
        supported = ("task_hint", "dir_name", "task_id", "generator_name", "generator", "domain")
        requested = [key for key in supported if key in include]
        if len(requested) != 1 or any(key not in supported for key in include):
            return None
        selector = requested[0]
        values = include[selector]
        if selector not in self.metadata.columns:
            return None
        if selector is None or values is None:
            return None
        if isinstance(values, (list, tuple, set)):
            task_hints = [str(v) for v in values]
        else:
            task_hints = [str(values)]
        cache = self._selector_split_indices.get(selector)
        if cache is None:
            cache = {
                (str(value), str(group_split)): [int(i) for i in group.index.tolist()]
                for (value, group_split), group in self.metadata.groupby([selector, "split"], sort=False)
            }
            self._selector_split_indices[selector] = cache
        out: list[int] = []
        for task_hint in task_hints:
            out.extend(cache.get((task_hint, str(split)), []))
        return out
