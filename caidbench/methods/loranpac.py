from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ..data.loader import build_dataloader
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, freeze_module, iter_limited_train_batches


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
        for _batch_idx, batch in iter_limited_train_batches(self, loader):
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
            return F.relu(x) if self.use_relu else x
        h = x.matmul(self.matrix.to(device=x.device, dtype=x.dtype))
        return F.relu(h) if self.use_relu else h


class OnlineTruncatedSVDSolver(nn.Module):
    """Incremental TSVD ridge solver following LoRanPAC's ``TSVDNet`` path."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        rank: int = 20000,
        ridge: float = 0.0,
        truncate_percent: float = 25.0,
        update_threshold: int = 10000,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.truncate_percent = float(truncate_percent)
        self.update_threshold = int(update_threshold)
        retained_dim = round(self.feature_dim * (1.0 - self.truncate_percent / 100.0))
        self.max_rank = min(int(retained_dim), self.rank)
        self.register_buffer("u", torch.empty(self.feature_dim, 0))
        self.register_buffer("s", torch.empty(0))
        self.register_buffer("cov_hy", torch.zeros(self.feature_dim, self.num_classes))
        self.register_buffer("svd_initialized", torch.tensor(False, dtype=torch.bool))
        self._feature_chunks: list[torch.Tensor] = []
        self.num_samples = 0

    def update(self, h: torch.Tensor, y: torch.Tensor) -> None:
        h = h.detach().float().to(self.cov_hy.device)
        y = y.long().to(self.cov_hy.device)
        self.cov_hy += h.t().matmul(_one_hot(y, self.num_classes, dtype=h.dtype))
        self.num_samples += int(h.shape[0])
        self._feature_chunks.append(h)
        current_batch_size = len(self._feature_chunks) * int(h.shape[0])
        if current_batch_size > self.update_threshold:
            self.update_svd()

    def _num_preserved(self) -> int:
        keep_by_samples = round(self.num_samples * (1.0 - self.truncate_percent / 100.0))
        return min(int(keep_by_samples), int(self.max_rank))

    def update_svd(self) -> None:
        if not self._feature_chunks:
            return
        features_h = torch.cat(self._feature_chunks, dim=0)
        self._feature_chunks = []
        num_preserved = self._num_preserved()
        if not bool(self.svd_initialized.item()):
            u, s, _vh = torch.linalg.svd(features_h.t(), full_matrices=False)
            self.u = u[:, :num_preserved].detach()
            self.s = s[:num_preserved].detach()
            self.svd_initialized.fill_(True)
            return

        upper_off_diag = self.u.t().matmul(features_h.t())
        residual = features_h.t() - self.u.matmul(upper_off_diag)
        q, r = torch.linalg.qr(residual, mode="reduced")
        lower = torch.cat(
            [
                torch.zeros(r.shape[0], self.s.shape[0], device=r.device, dtype=r.dtype),
                r,
            ],
            dim=1,
        )
        upper = torch.cat([torch.diag(self.s.to(device=r.device, dtype=r.dtype)), upper_off_diag], dim=1)
        basis_u, s, _vh = torch.linalg.svd(torch.cat([upper, lower], dim=0), full_matrices=False)
        basis_u = basis_u[:, :num_preserved]
        s = s[:num_preserved]
        updated_u = torch.cat([self.u.to(q.device), q], dim=1).matmul(basis_u)
        updated_u, _r = torch.linalg.qr(updated_u, mode="reduced")
        self.u = updated_u.detach()
        self.s = s.detach()

    def finalize(self) -> None:
        self.update_svd()

    def weight(self) -> torch.Tensor:
        if self.u.numel() == 0:
            return torch.zeros(self.num_classes, self.feature_dim, device=self.cov_hy.device)
        ut_cov = self.u.t().matmul(self.cov_hy)
        denom = self.s.square().unsqueeze(1) + float(self.ridge)
        return self.u.matmul(ut_cov / denom).t()


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
        tsvd_update_threshold: int = 10000,
        use_test_transform_for_tsvd: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(freeze_backbone=True, **kwargs)
        self.E = int(E)
        self.use_RE = bool(use_RE)
        self.coslinear = bool(coslinear)
        self.tsvd_batch_size = int(tsvd_batch_size)
        self.use_test_transform_for_tsvd = bool(use_test_transform_for_tsvd)
        feature_dim = int(self.detector.feature_dim)
        proj_dim = self.E if self.use_RE else feature_dim
        self.projector = RandomProjection(feature_dim, self.E if self.use_RE else 0, use_relu=use_relu)
        self.solver = OnlineTruncatedSVDSolver(
            proj_dim,
            self.num_classes,
            rank=rank,
            ridge=ridge,
            truncate_percent=truncate_percent,
            update_threshold=tsvd_update_threshold,
        )

    def _build_tsvd_loader(self, trainer: Any, task: Any, fallback_loader: Any) -> Any:
        if not self.use_test_transform_for_tsvd:
            return fallback_loader
        scenario = getattr(trainer, "scenario", None)
        if scenario is None or not hasattr(scenario, "source"):
            return fallback_loader
        task_index = None
        for idx, spec in enumerate(getattr(scenario, "tasks", [])):
            same_id = getattr(spec, "task_id", None) == getattr(task, "task_id", None)
            same_name = getattr(spec, "name", None) == getattr(task, "name", None)
            if spec is task or (same_id and same_name):
                task_index = idx
                break
        if task_index is None:
            return fallback_loader
        split_indices = getattr(scenario, "_split_indices", {})
        indices = split_indices.get((task_index, "train"))
        if indices is None:
            return fallback_loader
        transform = scenario._transform_for_split("test")
        dataset = scenario.source.make_dataset(
            indices,
            transform_cfg=transform,
            task_id=getattr(task, "task_id", task_index),
            task_name=getattr(task, "name", f"task{task_index}"),
        )
        return build_dataloader(
            dataset,
            batch_size=max(self.tsvd_batch_size, 1),
            shuffle=True,
            num_workers=int(getattr(trainer, "num_workers", 0)),
            drop_last=False,
        )

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        tsvd_loader = self._build_tsvd_loader(trainer, task, train_loader)
        was_training = self.training
        self.eval()
        task_samples = 0
        with torch.no_grad():
            for _batch_idx, batch in iter_limited_train_batches(trainer, tsvd_loader):
                batch = batch_to_device(batch, self.device)
                z = self.extract_features(batch["x"])
                h = self.projector(z)
                labels = _local_targets(batch["y"], self.num_classes).to(self.device)
                self.solver.update(h, labels)
                task_samples += int(labels.numel())
            self.solver.finalize()
        if was_training:
            self.train()
        dim = self.projector.out_dim if self.use_RE else int(self.detector.feature_dim)
        trainer.log_train_metrics(
            {
                "loranpac_rank": float(self.solver.s.numel()),
                "loranpac_samples": float(task_samples),
                "loranpac_dim": float(dim),
            },
            task=task,
            phase="solver",
        )
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
