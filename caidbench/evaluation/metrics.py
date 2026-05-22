from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray | list[Any]) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def binary_accuracy(logits: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray) -> float:
    logits_np = _to_numpy(logits)
    y_np = _to_numpy(y).astype(int)
    if logits_np.ndim == 2 and logits_np.shape[1] > 1:
        pred = logits_np.argmax(axis=1)
    else:
        pred = (logits_np.reshape(-1) > 0).astype(int)
    if y_np.size == 0:
        return float("nan")
    return float((pred.reshape(-1) == y_np.reshape(-1)).mean())


def multiclass_accuracy(logits: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray) -> float:
    logits_np = _to_numpy(logits)
    y_np = _to_numpy(y).astype(int)
    pred = logits_np.argmax(axis=1)
    if y_np.size == 0:
        return float("nan")
    return float((pred.reshape(-1) == y_np.reshape(-1)).mean())


def binary_auroc(logits: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray) -> float:
    logits_np = _to_numpy(logits)
    y_np = _to_numpy(y).astype(int).reshape(-1)
    if len(np.unique(y_np)) < 2:
        return float("nan")
    if logits_np.ndim == 2 and logits_np.shape[1] > 1:
        exp = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        score = exp[:, 1] / np.maximum(exp.sum(axis=1), 1e-12)
    else:
        score = 1.0 / (1.0 + np.exp(-logits_np.reshape(-1)))
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_np, score))
    except Exception:
        # Rank-based fallback equivalent to Mann-Whitney U / AUC.
        order = np.argsort(score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(score) + 1)
        pos = y_np == 1
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)


def expected_calibration_error(logits: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray, bins: int = 15) -> float:
    logits_np = _to_numpy(logits)
    y_np = _to_numpy(y).astype(int).reshape(-1)
    if logits_np.ndim == 1 or logits_np.shape[1] == 1:
        p1 = 1.0 / (1.0 + np.exp(-logits_np.reshape(-1)))
        conf = np.maximum(p1, 1 - p1)
        pred = (p1 >= 0.5).astype(int)
    else:
        exp = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    ece = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not mask.any():
            continue
        acc = (pred[mask] == y_np[mask]).mean()
        ece += mask.mean() * abs(acc - conf[mask].mean())
    return float(ece)


def summarize_logits(logits: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray, prefix: str = "") -> dict[str, float]:
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}acc": binary_accuracy(logits, y),
        f"{p}auc": binary_auroc(logits, y),
        f"{p}ece": expected_calibration_error(logits, y),
    }


@dataclass
class ContinualMetricMatrix:
    """Stores row-by-row performance after each incremental task.

    Rows represent the model state after training task i. Columns represent the
    evaluation task j. Values outside the lower triangle remain NaN.
    """

    task_names: list[str]
    acc: np.ndarray = field(init=False)
    auc: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.task_names)
        self.acc = np.full((n, n), np.nan, dtype=float)
        self.auc = np.full((n, n), np.nan, dtype=float)

    def update(self, train_index: int, eval_index: int, acc: float, auc: float) -> None:
        self.acc[train_index, eval_index] = acc
        self.auc[train_index, eval_index] = auc

    def average_accuracy(self, train_index: int | None = None, kind: str = "acc") -> float:
        matrix = self.acc if kind == "acc" else self.auc
        row = matrix[-1 if train_index is None else train_index]
        return float(np.nanmean(row))

    def average_forgetting(self, kind: str = "acc") -> float:
        matrix = self.acc if kind == "acc" else self.auc
        n = matrix.shape[0]
        vals: list[float] = []
        for j in range(max(n - 1, 0)):
            best_before = np.nanmax(matrix[: n - 1, j])
            final = matrix[n - 1, j]
            if not (math.isnan(best_before) or math.isnan(final)):
                vals.append(float(best_before - final))
        return float(np.mean(vals)) if vals else float("nan")

    def to_tables(self) -> dict[str, list[list[float | None]]]:
        def clean(m: np.ndarray) -> list[list[float | None]]:
            out = []
            for row in m.tolist():
                out.append([None if isinstance(v, float) and math.isnan(v) else float(v) for v in row])
            return out

        return {"acc": clean(self.acc), "auc": clean(self.auc)}
