from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..registry import register_method
from .base import ContinualMethod, batch_to_device, freeze_module


def _local_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = y.long()
    if y.numel() and (int(y.min()) < 0):
        raise ValueError("Labels must be non-negative for domain-incremental methods.")
    return torch.remainder(y, int(num_classes))


def _task_id(task: Any) -> int:
    return int(getattr(task, "task_id", task if isinstance(task, int) else 0))


def _one_hot(y: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return F.one_hot(y.long(), num_classes=int(num_classes)).to(dtype=dtype)


class FrozenFeatureMethod(ContinualMethod):
    """Common frozen-detector feature path for domain-incremental reproductions."""

    def __init__(self, freeze_backbone: bool = True, normalize_features: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.freeze_backbone = bool(freeze_backbone)
        self.normalize_features = bool(normalize_features)
        if self.freeze_backbone:
            freeze_module(self.detector.backbone)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_backbone:
            self.detector.backbone.eval()
        return self

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.detector.extract_features(x.to(self.device))
        z = z.float()
        return F.normalize(z, dim=-1) if self.normalize_features else z

    @torch.no_grad()
    def collect_features(self, loader: Any) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            features.append(self.extract_features(batch["x"]).detach().cpu())
            labels.append(_local_targets(batch["y"].detach().cpu(), self.num_classes))
        if was_training:
            self.train()
        if not features:
            return torch.empty(0, int(self.detector.feature_dim)), torch.empty(0, dtype=torch.long)
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)


class RandomProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_relu: bool = True) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.use_relu = bool(use_relu)
        if self.out_dim > 0:
            self.register_buffer("matrix", torch.randn(self.in_dim, self.out_dim))
        else:
            self.register_buffer("matrix", torch.empty(self.in_dim, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.out_dim <= 0:
            return x
        h = x.matmul(self.matrix.to(device=x.device, dtype=x.dtype))
        return F.relu(h) if self.use_relu else h


class OnlineTruncatedSVDSolver(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, rank: int = 20000, ridge: float = 0.0, truncate_percent: float = 25.0) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.truncate_percent = float(truncate_percent)
        self.register_buffer("u", torch.empty(self.feature_dim, 0))
        self.register_buffer("s", torch.empty(0))
        self.register_buffer("cov_hy", torch.zeros(self.feature_dim, self.num_classes))
        self.num_samples = 0

    def update(self, h: torch.Tensor, y: torch.Tensor) -> None:
        h = h.detach().float().to(self.cov_hy.device)
        y = y.long().to(self.cov_hy.device)
        self.cov_hy += h.t().matmul(_one_hot(y, self.num_classes, dtype=h.dtype))
        self.num_samples += int(h.shape[0])
        self._update_svd(h)

    def _update_svd(self, h: torch.Tensor) -> None:
        columns = h.t()
        if self.u.numel():
            summary = self.u * self.s.reshape(1, -1)
            columns = torch.cat([summary.to(columns.device), columns], dim=1)
        max_rank = max(1, min(self.rank, columns.shape[0], columns.shape[1]))
        keep_by_samples = int(round(self.num_samples * max(0.0, 1.0 - self.truncate_percent / 100.0)))
        keep = max(1, min(max_rank, keep_by_samples))
        u, s, _vh = torch.linalg.svd(columns, full_matrices=False)
        self.u = u[:, :keep].detach()
        self.s = s[:keep].detach()

    def weight(self) -> torch.Tensor:
        if self.u.numel() == 0:
            return torch.zeros(self.num_classes, self.feature_dim, device=self.cov_hy.device)
        ut_cov = self.u.t().matmul(self.cov_hy)
        denom = self.s.square().unsqueeze(1) + float(self.ridge)
        return self.u.matmul(ut_cov / denom.clamp_min(1e-12)).t()


@register_method("loranpac")
@register_method("lo_ranpac")
class LoRanPACMethod(FrozenFeatureMethod):
    """LoRanPAC low-rank random-feature ridge solver."""

    def __init__(
        self,
        E: int = 100000,
        rank: int = 20000,
        truncate_percent: float = 25.0,
        ridge: float = 0.0,
        use_RE: bool = True,
        use_relu: bool = True,
        coslinear: bool = False,
        tsvd_batch_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(freeze_backbone=True, **kwargs)
        self.E = int(E)
        self.use_RE = bool(use_RE)
        self.coslinear = bool(coslinear)
        self.tsvd_batch_size = int(tsvd_batch_size)
        feature_dim = int(self.detector.feature_dim)
        proj_dim = self.E if self.use_RE else feature_dim
        self.projector = RandomProjection(feature_dim, self.E if self.use_RE else 0, use_relu=use_relu)
        self.solver = OnlineTruncatedSVDSolver(proj_dim, self.num_classes, rank=rank, ridge=ridge, truncate_percent=truncate_percent)

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        features, labels = self.collect_features(train_loader)
        for start in range(0, features.shape[0], max(self.tsvd_batch_size, 1)):
            stop = start + max(self.tsvd_batch_size, 1)
            h = self.projector(features[start:stop].to(self.device))
            self.solver.update(h, labels[start:stop].to(self.device))
        dim = self.projector.out_dim if self.use_RE else int(self.detector.feature_dim)
        trainer.logger.info("task=%s loranpac_rank=%d samples=%d dim=%d", task.name, self.solver.s.numel(), labels.numel(), dim)
        trainer.log_metrics({"train/loranpac_rank": float(self.solver.s.numel()), "train/task_index": float(_task_id(task))})
        return True

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.extract_features(x)
        h = self.projector(z)
        w = self.solver.weight().to(device=h.device, dtype=h.dtype)
        if self.coslinear:
            logits = F.normalize(h, dim=-1).matmul(F.normalize(w, dim=-1).t())
        else:
            logits = h.matmul(w.t())
        return {"logits": logits, "features": z}
