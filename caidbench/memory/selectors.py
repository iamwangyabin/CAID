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


@torch.no_grad()
def extract_feature_logit_table(
    model,
    loader,
    device: torch.device | str,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    model.eval()
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
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
        logit = out["logits"] if isinstance(out, dict) else out
        features.append(z.detach().cpu())
        logits.append(logit.detach().cpu())
        labels.append(y.detach().cpu())
        rows.extend(split_batch(batch))
        if max_samples is not None and len(rows) >= max_samples:
            break
    if not features:
        empty = torch.empty(0)
        return empty, torch.empty(0, dtype=torch.long), empty, []
    feat = torch.cat(features, dim=0)[:max_samples]
    lab = torch.cat(labels, dim=0)[:max_samples]
    log = torch.cat(logits, dim=0)[:max_samples]
    return feat, lab, log, rows[:max_samples]


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


def dfil_official_indices(
    features: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    center_per_class: int,
    hard_per_class: int,
) -> list[int]:
    """DFIL memory selection: low distance-to-mean centers plus high-entropy hard samples."""
    if features.numel() == 0 or labels.numel() == 0:
        return []
    z = features.float().reshape(features.shape[0], -1)
    probs = F.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    selected: list[int] = []
    for c in labels.unique(sorted=True):
        idx = torch.where(labels == int(c))[0]
        if idx.numel() == 0:
            continue
        zc = z[idx]
        centroid = zc.mean(dim=0, keepdim=True)
        dist = torch.cdist(zc, centroid, p=2).squeeze(1)
        center_n = min(max(int(center_per_class), 0), idx.numel())
        center_local = torch.argsort(dist)[:center_n].tolist()
        used = set(center_local)
        remaining = [i for i in range(idx.numel()) if i not in used]
        hard_n = min(max(int(hard_per_class), 0), len(remaining))
        if hard_n > 0:
            rem_t = torch.tensor(remaining, dtype=torch.long)
            hard_order = torch.argsort(entropy[idx[rem_t]], descending=True)[:hard_n]
            hard_local = rem_t[hard_order].tolist()
        else:
            hard_local = []
        selected.extend(idx[center_local + hard_local].tolist())
    return selected


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


def _centered_alignment_scores(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.new_zeros(features.shape[0])
    z = F.normalize(features.float().reshape(features.shape[0], -1), dim=1)
    t = target.float().reshape(target.shape[0], -1)
    t = t - t.mean(dim=0, keepdim=True)
    k = z @ z.T
    l = t @ t.T
    n = z.shape[0]
    h = torch.eye(n, dtype=z.dtype, device=z.device) - torch.ones(n, n, dtype=z.dtype, device=z.device) / n
    kc = h @ k @ h
    lc = h @ l @ h
    return (kc * lc).sum(dim=1)


def _self_hsic_centrality_scores(features: torch.Tensor) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.new_zeros(features.shape[0])
    z = features.float().reshape(features.shape[0], -1)
    sq = torch.cdist(z, z, p=2).pow(2)
    vals = sq.detach().flatten()
    vals = vals[vals > 0]
    sigma = torch.sqrt(torch.median(vals)).item() if vals.numel() else 1.0
    sigma = max(float(sigma), 1e-6)
    k = torch.exp(-sq / (2.0 * sigma * sigma))
    n = z.shape[0]
    h = torch.eye(n, dtype=z.dtype, device=z.device) - torch.ones(n, n, dtype=z.dtype, device=z.device) / n
    kc = h @ k @ h
    return kc.pow(2).sum(dim=1)


def official_hgr_indices(features: torch.Tensor, labels: torch.Tensor, k: int, alpha: float = 0.5) -> list[int]:
    """Official-equivalent HGR selection with online-computed features.

    The released HGR buffer writer selects exemplars per class using a joint
    score over k-center coverage and HSIC centrality. This implementation keeps
    the same selection rule, but returns sample indices so CAIDBench can store
    image rows and recompute features online during replay.
    """
    if features.numel() == 0 or k <= 0:
        return []
    z = F.normalize(features.float().reshape(features.shape[0], -1), dim=1)
    selected: list[int] = []
    classes = labels.unique(sorted=True).tolist()
    base = k // max(len(classes), 1)
    quotas: dict[int, int] = {}
    remaining = k
    for c in classes:
        idx = torch.where(labels == int(c))[0]
        quotas[int(c)] = min(idx.numel(), base)
        remaining -= quotas[int(c)]
    for c in classes:
        if remaining <= 0:
            break
        idx = torch.where(labels == int(c))[0]
        extra = min(remaining, max(0, idx.numel() - quotas[int(c)]))
        quotas[int(c)] += extra
        remaining -= extra

    def _minmax(x: torch.Tensor) -> torch.Tensor:
        return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)

    for c in classes:
        idx = torch.where(labels == int(c))[0]
        quota = min(quotas[int(c)], idx.numel())
        if quota <= 0:
            continue
        feats = z[idx]
        centrality = _minmax(_self_hsic_centrality_scores(feats))
        centroid = feats.mean(dim=0, keepdim=True)
        d2_cent = (feats - centroid).pow(2).sum(dim=1)
        coverage = _minmax(d2_cent)
        score = float(alpha) * (1.0 - coverage) + (1.0 - float(alpha)) * (1.0 - centrality)
        class_selected = [int(torch.argmin(score).item())]
        min_d2 = (feats - feats[class_selected[0]]).pow(2).sum(dim=1)
        while len(class_selected) < quota:
            coverage = _minmax(min_d2)
            score = float(alpha) * (1.0 - coverage) + (1.0 - float(alpha)) * (1.0 - centrality)
            score[torch.tensor(class_selected, dtype=torch.long, device=score.device)] = float("inf")
            nxt = int(torch.argmin(score).item())
            class_selected.append(nxt)
            min_d2 = torch.minimum(min_d2, (feats - feats[nxt]).pow(2).sum(dim=1))
        selected.extend(idx[class_selected].tolist())
    return selected[: min(k, features.shape[0])]


def hsic_guided_indices(features: torch.Tensor, labels: torch.Tensor, nuisance: torch.Tensor | None, k: int, lambda_kc: float = 0.5) -> list[int]:
    """HSIC-Guided Replay: label relevance balanced with k-center coverage.

    The relevance term uses the per-sample contribution to centered kernel
    alignment with real/fake labels, and subtracts nuisance alignment when
    nuisance IDs are available. The coverage term is the current distance to
    the selected set, matching HGR's hybrid relevance plus k-center objective.
    """
    if features.numel() == 0 or k <= 0:
        return []
    n_classes = int(labels.max().item()) + 1 if labels.numel() else 1
    y = F.one_hot(labels.long(), num_classes=max(n_classes, 1)).float()
    relevance = _centered_alignment_scores(features, y)
    if nuisance is not None and nuisance.numel() == labels.numel():
        n_count = int(nuisance.max().item()) + 1 if nuisance.numel() else 1
        nuisance_oh = F.one_hot(nuisance.long(), num_classes=max(n_count, 1)).float()
        relevance = relevance - _centered_alignment_scores(features, nuisance_oh)
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min()).clamp_min(1e-12)
    z = F.normalize(features.float().reshape(features.shape[0], -1), dim=1)

    selected: list[int] = []
    classes = labels.unique(sorted=True).tolist()
    per_class = max(k // max(len(classes), 1), 1)
    for c in classes:
        idx = torch.where(labels == int(c))[0]
        if idx.numel() == 0:
            continue
        class_selected: list[int] = []
        quota = min(per_class, idx.numel())
        while len(class_selected) < quota:
            if class_selected:
                chosen = idx[torch.tensor(class_selected, dtype=torch.long, device=idx.device)]
                coverage = 1.0 - (z[idx] @ z[chosen].T).max(dim=1).values
            else:
                coverage = torch.ones(idx.numel(), dtype=z.dtype, device=z.device)
            coverage = (coverage - coverage.min()) / (coverage.max() - coverage.min()).clamp_min(1e-12)
            score = relevance[idx] + float(lambda_kc) * coverage
            if class_selected:
                score[torch.tensor(class_selected, dtype=torch.long, device=score.device)] = -float("inf")
            class_selected.append(int(torch.argmax(score).item()))
        selected.extend(idx[class_selected].tolist())
    if len(selected) < k:
        remaining = [i for i in kcenter_greedy(features, k, initial=selected) if i not in selected]
        selected.extend(remaining[: k - len(selected)])
    return selected[: min(k, features.shape[0])]
