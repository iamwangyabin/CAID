from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from torch.utils.data import Dataset

from .protocol import apply_filter, load_protocol, task_split_specs
from .sources import DataSource, build_data_source


def _resolve_protocol_index_path(protocol_ref: Any, index_path: Any) -> str:
    path = Path(str(index_path))
    if path.is_absolute():
        return str(path)
    if isinstance(protocol_ref, (str, Path)):
        return str(Path(protocol_ref).parent / path)
    return str(path)


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    domains: tuple[str, ...]
    generators: tuple[str, ...]
    scenes: tuple[str, ...]
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0


class ContinualScenario:
    """Ordered continual-learning scenario.

    The scenario is built from an Arrow-backed data source plus an optional
    protocol YAML. Storage and incremental task definitions stay decoupled:
    ``scenario.data`` selects Arrow/AID Arrow storage, while
    ``scenario.protocol`` selects task composition and order.
    """

    def __init__(
        self,
        *,
        source: DataSource,
        protocol: Mapping[str, Any] | str | Path | None = None,
        transform_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.transform_cfg = transform_cfg or {}
        self.source = source
        self.protocol = load_protocol(protocol)
        self._split_indices: dict[tuple[int, str], list[int]] = {}
        self.df = self.source.metadata.copy().reset_index(drop=True)
        self.tasks = self._build_protocol_tasks(self.protocol)

    @classmethod
    def from_config(cls, scenario_cfg: Mapping[str, Any]) -> "ContinualScenario":
        scfg = dict(scenario_cfg)
        if "order" in scfg:
            raise ValueError("scenario.order is no longer supported; reorder scenario.protocol.tasks instead")
        transform_cfg = scfg.get("transform")
        data_cfg = dict(scfg.get("data", {}))
        if not data_cfg and "backend" in scfg:
            data_cfg = {k: v for k, v in scfg.items() if k not in {"protocol", "transform"}}
        if not data_cfg:
            raise ValueError("Config must define scenario.data with an AID Arrow dataset directory")
        protocol_ref = scfg.get("protocol", {})
        protocol = load_protocol(protocol_ref)
        if "index_path" not in data_cfg and "index" not in data_cfg:
            protocol_index = protocol.get("index_path", protocol.get("index"))
            if protocol_index is not None:
                data_cfg["index_path"] = _resolve_protocol_index_path(protocol_ref, protocol_index)
        source = build_data_source(data_cfg)
        return cls(source=source, protocol=protocol, transform_cfg=transform_cfg)

    def _uniq(self, df: pd.DataFrame, col: str) -> tuple[str, ...]:
        if col not in df.columns or len(df) == 0:
            return tuple()
        return tuple(sorted(str(x) for x in df[col].dropna().unique().tolist()))

    def _build_protocol_tasks(self, protocol: Mapping[str, Any]) -> list[TaskSpec]:
        if "order" in protocol:
            raise ValueError("protocol.order is no longer supported; reorder the tasks list directly")
        # If no explicit tasks are provided, fall back to metadata.task_id.
        task_cfgs = list(protocol.get("tasks", []) or [])
        if not task_cfgs:
            if "task_id" not in self.df.columns:
                raise ValueError("No protocol.tasks provided and source metadata has no task_id")
            task_ids = sorted(int(x) for x in self.df["task_id"].dropna().unique().tolist())
            task_cfgs = [{"id": tid, "name": f"task{tid}", "filter": {"include": {"task_id": tid}}} for tid in task_ids]

        tasks: list[TaskSpec] = []
        for task_index, task_cfg in enumerate(task_cfgs):
            tid_raw = task_cfg.get("numeric_id", task_cfg.get("task_id", task_index))
            tid = int(tid_raw)
            name = str(task_cfg.get("name", task_cfg.get("id", f"task{tid}")))
            specs = task_split_specs(task_cfg)
            all_rows = []
            counts = {}
            for split, spec in specs.items():
                fast_select = getattr(self.source, "select_indices", None)
                idx = fast_select(spec) if callable(fast_select) else None
                if idx is None:
                    sdf = apply_filter(self.df, spec)
                    idx = [int(i) for i in sdf.index.tolist()]
                else:
                    sdf = self.df.iloc[idx]
                self._split_indices[(task_index, split)] = idx
                counts[split] = len(idx)
                if len(sdf):
                    all_rows.append(sdf)
            tdf = pd.concat(all_rows, axis=0) if all_rows else self.df.iloc[[]]
            tasks.append(
                TaskSpec(
                    task_id=tid,
                    name=name,
                    domains=self._uniq(tdf, "domain"),
                    generators=self._uniq(tdf, "generator"),
                    scenes=self._uniq(tdf, "scene"),
                    num_train=int(counts.get("train", 0)),
                    num_val=int(counts.get("val", 0)),
                    num_test=int(counts.get("test", 0)),
                )
            )
        return tasks

    def seen_dataset(self, split: str, upto_index: int, transform_split: str | None = None) -> Dataset:
        idx: list[int] = []
        for task_index in range(upto_index + 1):
            idx.extend(self._split_indices.get((task_index, split), []))
        return self.source.make_dataset(
            idx,
            transform_cfg=self._transform_for_split(transform_split or split),
            task_id=-1,
            task_name=f"seen_until_{upto_index}",
        )

    def task_dataset(self, split: str, task_index: int, transform_split: str | None = None) -> Dataset:
        task = self.tasks[task_index]
        idx = self._split_indices.get((task_index, split), [])
        return self.source.make_dataset(
            idx,
            transform_cfg=self._transform_for_split(transform_split or split),
            task_id=task.task_id,
            task_name=task.name,
        )

    def _transform_for_split(self, split: str) -> Any:
        cfg = self.transform_cfg
        if not isinstance(cfg, Mapping):
            return cfg
        if split in cfg:
            return cfg[split]
        if split == "val" and "test" in cfg:
            return cfg["test"]
        if split in {"val", "test"} and "eval" in cfg:
            return cfg["eval"]
        if "default" in cfg:
            return cfg["default"]
        return cfg
