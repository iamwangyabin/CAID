from __future__ import annotations

"""AID-style Arrow packaging utilities.

Design principle:
  - Arrow/HuggingFace dataset stores image bytes only: column `image`.
  - Sample metadata lives in lightweight sidecar files:
      mapping.json      path -> row index in Arrow dataset
      index.jsonl       one metadata record per sample
      train/val/test.json  AID-compatible subset dictionaries
  - Continual task construction is NOT baked into Arrow. YAML protocols filter
    sidecar metadata to define tasks and task order.
"""

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AID_IMAGE_COLUMN = "image"
AID_MAPPING_FILE = "mapping.json"
AID_INDEX_FILE = "index.jsonl"  # legacy CAID metadata sidecar; optional, not required by AID
AID_META_FILE = "caid_meta.jsonl"  # preferred CAID metadata sidecar; AID ignores it
AID_INFO_FILE = "caid_info.json"

# Keep this name for compatibility with older code/tests, but it now means the
# sidecar index schema rather than Arrow table columns.
CANONICAL_COLUMNS = [
    "path",
    "label",
    "split",
    "dataset",
    "domain",
    "generator",
    "manipulation",
    "video_id",
    "frame_idx",
    "scene",
    "task_hint",
    "preprocess_profile",
]

DEFAULTS: dict[str, Any] = {
    "path": "",
    "label": -1,
    "split": "train",
    "dataset": "unknown",
    "domain": "unknown",
    "generator": "unknown",
    "manipulation": "unknown",
    "video_id": "",
    "frame_idx": -1,
    "scene": "unknown",
    "task_hint": "",
    "preprocess_profile": "",
}

STRING_COLUMNS = [c for c in CANONICAL_COLUMNS if c not in {"label", "frame_idx"}]
INT_COLUMNS = ["label", "frame_idx"]


def read_file_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def relpath(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def _resolve_path(raw: str | Path, root: Path | None) -> Path:
    p = Path(str(raw))
    if root is not None and not p.is_absolute():
        p = root / p
    return p


def normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    root: str | Path | None = None,
    strict_images: bool = False,
    **_: Any,
) -> pd.DataFrame:
    """Normalize arbitrary scanner/manifest records into a minimal sidecar index.

    The returned DataFrame intentionally contains only lightweight metadata.  It
    is NOT the Arrow payload.  The payload is written by :func:`write_aid_dataset`
    as a one-column HF Dataset with column `image`.
    """
    root_path = Path(root) if root else None
    rows: list[dict[str, Any]] = []
    for rec0 in records:
        rec = dict(rec0)
        row = dict(DEFAULTS)
        for key in CANONICAL_COLUMNS:
            if key in rec and rec[key] is not None:
                row[key] = rec[key]

        raw_path = rec.get("path") or rec.get("image_path") or rec.get("file") or rec.get("filepath")
        if raw_path in (None, ""):
            raise ValueError("AID-style Arrow records require an image path")
        abs_path = _resolve_path(raw_path, root_path)
        if strict_images and (not abs_path.exists() or abs_path.suffix.lower() not in IMAGE_EXTENSIONS):
            raise FileNotFoundError(f"Missing/unsupported image path: {abs_path}")
        row["path"] = relpath(abs_path, root_path)

        if row["domain"] in (None, "", "unknown"):
            row["domain"] = row["dataset"] if row["dataset"] not in (None, "", "unknown") else row["generator"]
        if row["generator"] in (None, "", "unknown"):
            row["generator"] = row["manipulation"] if row["manipulation"] not in (None, "", "unknown") else row["domain"]
        if row["manipulation"] in (None, ""):
            row["manipulation"] = "unknown"
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULTS[col]
    df = df[CANONICAL_COLUMNS]
    for col in STRING_COLUMNS:
        df[col] = df[col].fillna(DEFAULTS[col]).astype(str)
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULTS[col]).astype("int64")
    return df.reset_index(drop=True)


def _iter_image_bytes(df: pd.DataFrame, root: str | Path | None = None) -> list[bytes]:
    root_path = Path(root) if root else None
    images: list[bytes] = []
    for p in df["path"].tolist():
        img_path = _resolve_path(str(p), root_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image referenced by sidecar does not exist: {img_path}")
        images.append(read_file_bytes(img_path))
    return images


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _subset_name(prefix: str, value: Any) -> str:
    text = str(value).replace("/", "_").replace(" ", "_")
    return f"{prefix}:{text}"


def _split_payload(df: pd.DataFrame, split: str) -> dict[str, dict[str, int]]:
    """Build an AID-compatible split JSON payload.

    AID's ArrowDatasets expects `<split>.json` to be a dictionary:
        {subset_name: {relative_image_path: binary_label}}

    We keep this exact shape.  For CAID protocols we also add convenient
    subset names, but they are still plain AID subsets, not Arrow columns.
    """
    part = df[df["split"].astype(str) == split]
    payload: dict[str, dict[str, int]] = {"all": {}}
    for _, row in part.iterrows():
        path = str(row["path"])
        label = int(row["label"])
        payload["all"][path] = label
        payload.setdefault("real" if label == 0 else "fake", {})[path] = label
        for col, prefix in [
            ("dataset", "dataset"),
            ("domain", "domain"),
            ("generator", "generator"),
            ("manipulation", "manipulation"),
            ("task_hint", "task"),
        ]:
            val = row.get(col, "")
            if val is not None and str(val) not in {"", "unknown", "nan"}:
                # Namespaced subsets avoid collisions and are used by CAID.
                payload.setdefault(_subset_name(prefix, val), {})[path] = label
                # Direct subsets match AID configs, which often use values such
                # as `car`, `chair`, `sd14`, or `Deepfakes` directly.
                payload.setdefault(str(val), {})[path] = label
    return payload


def write_sidecars(df: pd.DataFrame, out_dir: str | Path, *, info: Mapping[str, Any] | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # EXACT AID core sidecar: relative image path -> row index in the HF dataset.
    mapping = {str(path): int(i) for i, path in enumerate(df["path"].tolist())}
    with open(out / AID_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Optional CAID metadata. AID ignores these files; CAID uses them only for
    # richer YAML protocol filtering.  Keep index.jsonl for backward compat, but
    # do not require it for loading AID datasets.
    records = df.to_dict(orient="records")
    for meta_name in (AID_META_FILE, AID_INDEX_FILE):
        with open(out / meta_name, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    splits = sorted(set(df["split"].astype(str).tolist()) | {"train", "val", "test"})
    for split in splits:
        payload = _split_payload(df, split)
        with open(out / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        # AID configs often use `train_binary` for binary real/fake training.
        if split in {"train", "test", "val"}:
            with open(out / f"{split}_binary.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
    meta = {
        "format": "aid-arrow-compatible",
        "arrow_columns": [AID_IMAGE_COLUMN],
        "aid_core_sidecars": [AID_MAPPING_FILE, "<split>.json"],
        "optional_caid_sidecars": [AID_META_FILE, AID_INDEX_FILE, AID_INFO_FILE],
        "index_columns": CANONICAL_COLUMNS,
        "num_rows": int(len(df)),
    }
    if info:
        meta.update(dict(info))
    with open(out / AID_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_aid_dataset(
    df: pd.DataFrame,
    out: str | Path,
    *,
    root: str | Path | None = None,
    image_column: str = AID_IMAGE_COLUMN,
    info: Mapping[str, Any] | None = None,
) -> Path:
    """Write AID-compatible HF/Arrow dataset.

    Output directory layout:
      out/
        data-*.arrow, dataset_info.json, state.json   # from datasets.save_to_disk
        mapping.json                                  # path -> row index
        index.jsonl                                   # metadata sidecar
        train.json / val.json / test.json             # AID-style subset files
        caid_info.json                                # schema note
    """
    try:
        from datasets import Dataset
    except Exception as e:  # pragma: no cover
        raise ImportError("AID-style Arrow output requires: pip install -e '.[arrow]'") from e
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    images = _iter_image_bytes(df, root=root)
    ds = Dataset.from_dict({image_column: images})
    ds.save_to_disk(str(out_path))
    write_sidecars(df, out_path, info=info)
    return out_path


def write_arrow_table(
    df: pd.DataFrame,
    out: str | Path,
    *,
    fmt: str = "aid",
    root: str | Path | None = None,
) -> Path:
    """Write dataset in AID-style format by default.

    `fmt=aid` is the intended format.  `arrow`/`parquet` remain only for compact
    metadata debugging and are not used by the training configs.
    """
    fmt = fmt.lower()
    out_path = Path(out)
    if fmt in {"aid", "hf", "hf_dataset", "datasets", "dataset"}:
        return write_aid_dataset(df, out_path, root=root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise ImportError("Arrow writing requires optional dependency: pip install -e '.[arrow]'") from e
    table = pa.Table.from_pandas(df, preserve_index=False)
    if fmt == "parquet" or out_path.suffix.lower() == ".parquet":
        pq.write_table(table, out_path)
    else:
        with pa.OSFile(str(out_path), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    return out_path


def read_index_sidecar(path: str | Path) -> pd.DataFrame:
    root = Path(path)
    if root.is_file():
        root = root.parent
    index_path = root / AID_INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"AID sidecar not found: {index_path}")
    return normalize_records(_read_jsonl(index_path))


def read_any_table_to_df(path: str | Path) -> pd.DataFrame:
    """Debug helper: return sidecar index for AID datasets, else Arrow table."""
    path = Path(path)
    if path.is_dir() and (path / AID_INDEX_FILE).exists():
        return read_index_sidecar(path)
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    import pyarrow as pa

    if path.suffix.lower() == ".parquet":
        return pq.read_table(path).to_pandas()
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_file(source).read_all().to_pandas()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_stream(source).read_all().to_pandas()


def read_caid_meta_sidecar(path: str | Path) -> pd.DataFrame | None:
    """Read optional CAID metadata sidecar if present.

    AID itself does not need this file. It is used only to recover fields such
    as dataset/generator/video_id for flexible CAID protocol YAMLs.
    """
    root = Path(path)
    if root.is_file():
        root = root.parent
    for name in (AID_META_FILE, AID_INDEX_FILE):
        p = root / name
        if p.exists():
            return normalize_records(_read_jsonl(p))
    return None


def _iter_aid_split_files(root: Path) -> list[Path]:
    skip = {AID_MAPPING_FILE, AID_INFO_FILE, AID_META_FILE, AID_INDEX_FILE, "dataset_info.json", "state.json", "dataset_dict.json"}
    return sorted(p for p in root.glob("*.json") if p.name not in skip)


def read_aid_split_sidecars(path: str | Path) -> pd.DataFrame:
    """Read AID-native split sidecars without requiring any CAID metadata.

    This is the compatibility path for datasets already processed by AID.  It
    follows AID's loader contract:

      - ``load_from_disk(data_root)`` reads a HF/Arrow dataset whose samples are
        indexed by row id.
      - ``mapping.json`` maps ``relative_image_path -> row_id``.
      - ``<split>.json`` stores ``{subset_name: {relative_image_path: label}}``
        or, for some AID pair datasets, a list of image paths.

    CAIDBench constructs one metadata row for each ``(split, path)`` pair.  If a
    path belongs to multiple AID subsets inside the same split, the subset names
    are stored as a semicolon-separated membership string in both ``subset`` and
    ``task_hint``.  This lets protocol YAMLs select existing AID subsets directly,
    e.g. ``filter: {subset: sd15}``, without rebuilding or modifying the Arrow
    dataset.
    """
    root = Path(path)
    if root.is_file():
        root = root.parent
    mapping_path = root / AID_MAPPING_FILE
    if not mapping_path.exists():
        raise FileNotFoundError(f"AID mapping sidecar not found: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = {str(k): int(v) for k, v in json.load(f).items()}

    meta_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    split_files = _iter_aid_split_files(root)
    if not split_files:
        raise FileNotFoundError(
            f"No AID split JSON files found under {root}. Expected files such as train.json/test.json."
        )

    for sf in split_files:
        split_name = sf.stem
        with open(sf, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Some AID variants store a split as a plain list of image paths.  The
        # label is then not encoded in the split file; keep it as -1 unless a
        # richer CAID metadata sidecar later provides the value.
        if isinstance(payload, list):
            for p in payload:
                p = str(p)
                if p not in mapping:
                    continue
                key = (split_name, p)
                rec = meta_by_key.setdefault(
                    key,
                    {"path": p, "label": -1, "split": split_name, "subsets": set(), "_rowid": int(mapping[p])},
                )
                rec["subsets"].add("all")
            continue

        if not isinstance(payload, dict):
            continue

        for subset, subset_items in payload.items():
            subset = str(subset)
            if isinstance(subset_items, dict):
                iterator = subset_items.items()
            elif isinstance(subset_items, list):
                # AID sometimes uses {"real": [paths], "fake": [paths]}.
                lname = subset.lower()
                inferred = 0 if lname == "real" else 1 if lname == "fake" else -1
                iterator = ((p, inferred) for p in subset_items)
            else:
                continue

            for p, label in iterator:
                p = str(p)
                if p not in mapping:
                    # The AID loader would fail later too; skip here so a broken
                    # subset does not poison unrelated experiments.
                    continue
                key = (split_name, p)
                rec = meta_by_key.setdefault(
                    key,
                    {"path": p, "label": -1, "split": split_name, "subsets": set(), "_rowid": int(mapping[p])},
                )
                rec["subsets"].add(subset)
                try:
                    rec["label"] = int(label)
                except Exception:
                    pass

    rows: list[dict[str, Any]] = []
    for (_split, _path), rec0 in sorted(meta_by_key.items(), key=lambda kv: (kv[0][0], mapping.get(kv[0][1], 10**18))):
        rec = dict(rec0)
        subsets = rec.pop("subsets", set())
        subset_text = ";".join(sorted(str(s) for s in subsets))
        rec["subset"] = subset_text
        rec["task_hint"] = subset_text
        rec.setdefault("dataset", "unknown")
        rec.setdefault("domain", "unknown")
        rec.setdefault("generator", "unknown")
        rec.setdefault("manipulation", "unknown")
        rows.append(rec)

    if not rows:
        raise ValueError(f"AID sidecars under {root} did not reference any paths present in mapping.json")

    df = normalize_records(rows)
    # normalize_records keeps only canonical columns, so restore AID-specific
    # fields required by the protocol layer.
    row_lookup = {(r["split"], r["path"]): r for r in rows}
    df["_rowid"] = [int(row_lookup[(str(split), str(path))]["_rowid"]) for split, path in zip(df["split"], df["path"])]
    df["subset"] = [str(row_lookup[(str(split), str(path))].get("subset", "")) for split, path in zip(df["split"], df["path"])]
    df["aid_subset"] = df["subset"]
    return df.sort_values(["split", "_rowid"]).reset_index(drop=True)
