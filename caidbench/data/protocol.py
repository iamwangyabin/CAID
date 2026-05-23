from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..config import load_yaml

_SPLITS = ("train", "val", "test")
_RESERVED_FILTER_KEYS = {"include", "exclude", "query", "where", "split", "limit", "sample", "seed"}


def load_protocol(protocol: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    """Load a protocol definition.

    A protocol is the experiment-level mapping from dataset metadata to
    continual tasks. It deliberately lives outside Arrow storage.
    """
    if protocol is None:
        return {}
    if isinstance(protocol, Mapping):
        return dict(protocol)
    return load_yaml(protocol)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _series_in(series: pd.Series, values: Any) -> pd.Series:
    vals = _as_list(values)
    if not vals:
        return pd.Series([True] * len(series), index=series.index)
    # Compare as strings for metadata robustness, except numeric labels.
    if pd.api.types.is_numeric_dtype(series):
        return series.isin(vals)
    return series.astype(str).isin([str(v) for v in vals])


def _series_membership_in(series: pd.Series, values: Any) -> pd.Series:
    """Match scalar or semicolon-separated membership strings.

    AID split files select samples by subset names, while a sample can belong to
    multiple subsets such as ``all;fake;sd15``.  Protocol YAML can therefore use
    ``subset: sd15`` and this helper will check membership rather than exact full
    string equality.
    """
    vals = {str(v) for v in _as_list(values)}
    if not vals:
        return pd.Series([True] * len(series), index=series.index)

    def has_any(x: Any) -> bool:
        parts = {p for p in str(x).split(";") if p != ""}
        return bool(parts & vals) or str(x) in vals

    return series.map(has_any)


def apply_filter(df: pd.DataFrame, spec: Mapping[str, Any] | None) -> pd.DataFrame:
    """Apply a small YAML-friendly filter DSL to a metadata DataFrame.

    Supported forms:
      filter:
        include: {dataset: [FF++], generator: [Deepfakes]}
        exclude: {split: [val]}
        query: "label == 1 and frame_idx < 32"
        split: train
        limit: 1000

    For convenience, top-level keys other than include/exclude/query/split are
    treated as include filters, e.g. {dataset: FF++, split: train}.
    """
    if spec is None:
        return df
    spec = dict(spec)
    mask = pd.Series([True] * len(df), index=df.index)

    # Top-level split shortcut.
    if "split" in spec:
        if "split" not in df.columns:
            raise ValueError("Filter requested split but source metadata has no split column")
        mask &= _series_in(df["split"], spec["split"])

    include = dict(spec.get("include", {}) or {})
    for k, v in spec.items():
        if k not in _RESERVED_FILTER_KEYS:
            include[k] = v
    for key, values in include.items():
        lookup_key = key
        if key in {"subset", "subsets", "aid_subset"} and key not in df.columns:
            lookup_key = "subset" if "subset" in df.columns else "task_hint"
        if lookup_key not in df.columns:
            raise ValueError(f"Filter includes unknown metadata column: {key}")
        if lookup_key in {"subset", "subsets", "aid_subset", "task_hint"}:
            mask &= _series_membership_in(df[lookup_key], values)
        else:
            mask &= _series_in(df[lookup_key], values)

    exclude = dict(spec.get("exclude", {}) or {})
    for key, values in exclude.items():
        lookup_key = key
        if key in {"subset", "subsets", "aid_subset"} and key not in df.columns:
            lookup_key = "subset" if "subset" in df.columns else "task_hint"
        if lookup_key not in df.columns:
            raise ValueError(f"Filter excludes unknown metadata column: {key}")
        if lookup_key in {"subset", "subsets", "aid_subset", "task_hint"}:
            mask &= ~_series_membership_in(df[lookup_key], values)
        else:
            mask &= ~_series_in(df[lookup_key], values)

    query = spec.get("query", spec.get("where"))
    out = df[mask]
    if query:
        out = out.query(str(query), engine="python")

    seed = int(spec.get("seed", 0))
    sample = spec.get("sample")
    limit = spec.get("limit")
    if sample is not None:
        n = int(sample)
        out = out.sample(n=min(n, len(out)), random_state=seed)
    elif limit is not None:
        out = out.iloc[: int(limit)]
    return out


def split_filter(base: Mapping[str, Any] | None, split: str) -> dict[str, Any]:
    out = dict(base or {})
    out.setdefault("split", split)
    return out


def task_split_specs(task_cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize per-task train/val/test filters.

    A task may either provide explicit split filters:
      train: {split: train, include: {...}}
      test:  {split: test, include: {...}}

    or provide a base filter and let this function add split=train/val/test:
      filter: {include: {dataset: FF++}}
    """
    base = task_cfg.get("filter", task_cfg.get("include"))
    if base is not None and "include" not in base and "filter" not in task_cfg:
        # task.include is a shortcut for filter.include
        base = {"include": base}
    specs: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        if split in task_cfg and task_cfg[split] is not None:
            specs[split] = dict(task_cfg[split] or {})
        else:
            specs[split] = split_filter(base, split)
    return specs
