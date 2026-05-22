from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .replay import split_batch


@torch.no_grad()
def collect_batch_rows(loader, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in loader:
        rows.extend(split_batch(batch))
        if max_samples is not None and len(rows) >= max_samples:
            return rows[:max_samples]
    return rows


@torch.no_grad()
def extract_feature_table(model, loader, device: torch.device | str, max_samples: int | None = None) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        out = model.predict(batch) if hasattr(model, "predict") else model(x)
        z = out.get("features") if isinstance(out, dict) else None
        if z is None and hasattr(model, "detector"):
            z = model.detector.extract_features(x)
        if z is None:
            z = out["logits"] if isinstance(out, dict) else out
        features.append(z.detach().cpu())
        labels.append(y.detach().cpu())
        rows.extend(split_batch(batch))
        if max_samples is not None and len(rows) >= max_samples:
            break
    if not features:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []
    feat = torch.cat(features, dim=0)[:max_samples]
    lab = torch.cat(labels, dim=0)[:max_samples]
    return feat, lab, rows[:max_samples]


def random_indices(n: int, k: int, seed: int = 0) -> list[int]:
    if k <= 0 or n <= 0:
        return []
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n), size=min(k, n), replace=False).tolist()


def kcenter_greedy(features: torch.Tensor, k: int, seed: int = 0, initial: list[int] | None = None) -> list[int]:
    if features.numel() == 0 or k <= 0:
        return []
    n = features.shape[0]
    z = F.normalize(features.float(), dim=1)
    selected = list(initial or [])[:k]
    if not selected:
        selected = [int(np.random.default_rng(seed).integers(0, n))]
    dist = torch.full((n,), float("inf"))
    for idx in selected:
        d = 1.0 - (z @ z[idx].view(-1, 1)).squeeze(1)
        dist = torch.minimum(dist, d.cpu())
    while len(selected) < min(k, n):
        idx = int(torch.argmax(dist).item())
        selected.append(idx)
        d = 1.0 - (z @ z[idx].view(-1, 1)).squeeze(1)
        dist = torch.minimum(dist, d.cpu())
    return selected[: min(k, n)]


def central_and_hard_indices(features: torch.Tensor, labels: torch.Tensor, k: int, hard_ratio: float = 0.5) -> list[int]:
    """DFIL-style central + hard exemplar heuristic."""
    if features.numel() == 0 or k <= 0:
        return []
    z = F.normalize(features.float(), dim=1)
    selected: list[int] = []
    classes = labels.unique(sorted=True).tolist()
    per_class = max(k // max(len(classes), 1), 1)
    for c in classes:
        idx = torch.where(labels == int(c))[0]
        if idx.numel() == 0:
            continue
        zc = z[idx]
        centroid = F.normalize(zc.mean(dim=0, keepdim=True), dim=1)
        sim = (zc @ centroid.t()).squeeze(1)
        n_hard = min(int(round(per_class * hard_ratio)), idx.numel())
        n_center = min(per_class - n_hard, idx.numel() - n_hard)
        center_local = torch.topk(sim, k=max(n_center, 0), largest=True).indices.tolist() if n_center > 0 else []
        hard_local = torch.topk(sim, k=max(n_hard, 0), largest=False).indices.tolist() if n_hard > 0 else []
        selected.extend(idx[center_local + hard_local].tolist())
    if len(selected) < k:
        for i in kcenter_greedy(features, k - len(selected), initial=selected):
            if i not in selected:
                selected.append(i)
    return selected[: min(k, features.shape[0])]


def sparse_uniform_indices(features: torch.Tensor, k: int, stability: torch.Tensor | None = None) -> list[int]:
    """SUR-LID-inspired sparse uniform selection.

    Stable samples are preferred, then k-center spread enforces uniform coverage.
    """
    if features.numel() == 0 or k <= 0:
        return []
    if stability is None:
        stability = torch.linalg.norm(features.float(), dim=1)
    keep = torch.topk(stability, k=min(max(k * 3, k), features.shape[0]), largest=True).indices
    sub = features[keep]
    local = kcenter_greedy(sub, k)
    return keep[local].tolist()


def hsic_guided_indices(features: torch.Tensor, labels: torch.Tensor, nuisance: torch.Tensor | None, k: int) -> list[int]:
    """HSIC-Guided Replay proxy: preserve class balance and feature coverage.

    If nuisance IDs are available, allocate across class x nuisance cells before
    running k-center inside each cell. This mirrors the intent of reducing
    generator/identity collapse without requiring paper-specific scores.
    """
    if features.numel() == 0 or k <= 0:
        return []
    if nuisance is None:
        return central_and_hard_indices(features, labels, k, hard_ratio=0.0)
    cells = []
    for c in labels.unique(sorted=True):
        for n in nuisance.unique(sorted=True):
            idx = torch.where((labels == c) & (nuisance == n))[0]
            if idx.numel() > 0:
                cells.append(idx)
    per = max(k // max(len(cells), 1), 1)
    selected: list[int] = []
    for idx in cells:
        local = kcenter_greedy(features[idx], per)
        selected.extend(idx[local].tolist())
    if len(selected) < k:
        remaining = [i for i in kcenter_greedy(features, k) if i not in selected]
        selected.extend(remaining[: k - len(selected)])
    return selected[: min(k, features.shape[0])]
