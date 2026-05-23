from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd


_LABEL_COLUMNS = ("object_labels", "topk_object_labels", "object_label", "topk_labels")
_SCORE_COLUMNS = ("object_scores", "topk_object_scores", "object_score", "topk_scores")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _parse_sequence(value: Any) -> list[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw.split(",") if part.strip()]
        return _parse_sequence(parsed)
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_label_score(item: Any, default_score: float = 1.0) -> tuple[str, float]:
    if isinstance(item, Mapping):
        label = item.get("label", item.get("name", item.get("class", item.get("object", ""))))
        score = item.get("score", item.get("confidence", item.get("prob", default_score)))
        return str(label), float(score)
    if isinstance(item, (list, tuple)) and item:
        label = item[0]
        score = item[1] if len(item) > 1 else default_score
        return str(label), float(score)
    if isinstance(item, str) and ":" in item:
        label, score = item.rsplit(":", 1)
        try:
            return label.strip(), float(score)
        except ValueError:
            return item.strip(), default_score
    return str(item), default_score


def parse_object_labels(row: Mapping[str, Any]) -> list[tuple[str, float]] | None:
    """Parse optional top-k object labels from Arrow metadata.

    Accepted forms:
      - object_labels='[["person", 92.0], ["face", 88.0]]'
      - object_labels='person:92,face:88'
      - object_labels='person,face' plus object_scores='92,88'
      - topk_object_labels / topk_object_scores aliases.
    """

    label_value = None
    for col in _LABEL_COLUMNS:
        if col in row and not _is_missing(row[col]):
            label_value = row[col]
            break
    if label_value is None:
        return None

    labels = _parse_sequence(label_value)
    if not labels:
        return None

    score_value = None
    for col in _SCORE_COLUMNS:
        if col in row and not _is_missing(row[col]):
            score_value = row[col]
            break
    scores = _parse_sequence(score_value)

    out: list[tuple[str, float]] = []
    if scores and not any(isinstance(item, (list, tuple, Mapping)) for item in labels):
        for label, score in zip(labels, scores):
            out.append((str(label), float(score)))
    else:
        out = [_as_label_score(item) for item in labels]
    return [(label, score) for label, score in out if label]
