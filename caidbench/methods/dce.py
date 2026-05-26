from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.covariance import OAS

from ..registry import register_method
from .base import ContinualMethod, batch_to_device, build_optimizer, freeze_module


def _local_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = y.long()
    if y.numel() and (int(y.min()) < 0):
        raise ValueError("Labels must be non-negative for official DIL methods.")
    return torch.remainder(y, int(num_classes))


def _task_id(task: Any) -> int:
    return int(getattr(task, "task_id", task if isinstance(task, int) else 0))


def _official_optimizer_cfg(
    optimizer_cfg: dict[str, Any] | None,
    task_id: int,
    *,
    init_lr: float | None = None,
    lr: float | None = None,
    lrate: float | None = None,
    init_weight_decay: float | None = None,
    weight_decay: float | None = None,
    optimizer_type: str = "sgd",
) -> dict[str, Any]:
    cfg = dict(optimizer_cfg or {})
    cfg.setdefault("type", optimizer_type)
    cfg.setdefault("momentum", 0.9)
    official_lr = init_lr if int(task_id) == 0 and init_lr is not None else lr if lr is not None else lrate
    official_wd = init_weight_decay if int(task_id) == 0 and init_weight_decay is not None else weight_decay
    if official_lr is not None:
        cfg["lr"] = float(official_lr)
    if official_wd is not None:
        cfg["weight_decay"] = float(official_wd)
    return cfg


def _as_int_list(value: Sequence[int] | None) -> list[int]:
    return [int(v) for v in value] if value is not None else []


def _official_task_epochs(trainer: Any, task_id: int, init_epoch: int | None, epochs: int | None) -> int:
    if int(task_id) == 0 and init_epoch is not None:
        return max(int(init_epoch), 1)
    if epochs is not None:
        return max(int(epochs), 1)
    return max(int(getattr(trainer, "max_epochs", 1)), 1)


def _official_scheduler(
    optimizer: torch.optim.Optimizer,
    task_id: int,
    *,
    init_milestones: Sequence[int] | None = None,
    milestones: Sequence[int] | None = None,
    init_lr_decay: float | None = None,
    lr_decay: float | None = None,
    lrate_decay: float | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    points = _as_int_list(init_milestones if int(task_id) == 0 and init_milestones is not None else milestones)
    gamma = init_lr_decay if int(task_id) == 0 and init_lr_decay is not None else lrate_decay if lrate_decay is not None else lr_decay
    if not points or gamma is None:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=points, gamma=float(gamma))


def _run_minibatch_loop(
    method: ContinualMethod,
    trainer: Any,
    task: Any,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    method.train()
    for epoch in range(int(epochs)):
        totals: dict[str, float] = {}
        n = 0
        for batch in train_loader:
            out = method.observe(batch, task)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            method.transform_gradients(task)
            if trainer.grad_clip:
                torch.nn.utils.clip_grad_norm_(method.parameters(), trainer.grad_clip)
            optimizer.step()
            trainer.advance_step()
            method.after_optimizer_step(task)
            for key, value in out.items():
                if key == "logits":
                    continue
                if torch.is_tensor(value) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            n += 1
        if scheduler is not None:
            scheduler.step()
        if totals:
            metrics = {key: value / max(n, 1) for key, value in totals.items()}
            msg = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            trainer.logger.info("task=%s epoch=%d/%d %s", task.name, epoch + 1, epochs, msg)
            trainer.log_metrics(
                {
                    **{f"train/{key}": value for key, value in metrics.items()},
                    "train/task_index": float(_task_id(task)),
                    "train/epoch": epoch + 1,
                }
            )


class FrozenFeatureMethod(ContinualMethod):
    """Common frozen-detector feature path for official DIL reproductions."""

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


class CosineLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sigma: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.sigma = nn.Parameter(torch.ones(1)) if sigma else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
        return logits * self.sigma if self.sigma is not None else logits


class DCEExpert(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, max(dim // 2, 1)))
        self.head = CosineLinear(max(dim // 2, 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.float()))


class DCESelector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(in_dim), int(hidden_dim)), nn.ReLU(inplace=True), nn.Linear(int(hidden_dim), int(out_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


@register_method("dce")
class DCEMethod(FrozenFeatureMethod):
    """Dual-Balance Collaborative Experts for imbalanced DIL."""

    def __init__(
        self,
        total_sessions: int = 7,
        bal_epoch: int = 10,
        selector_epoch: int = 10,
        selector_lr: float = 0.01,
        num_sampled_pcls: int = 256,
        use_sm: bool = False,
        margin_sample_num: int = 10,
        use_oas_covariance: bool = True,
        feature_scaling_mode: int = 0,
        init_lr: float | None = None,
        lr: float | None = None,
        lrate: float | None = None,
        init_weight_decay: float | None = None,
        weight_decay: float | None = None,
        optimizer_type: str = "sgd",
        init_epoch: int | None = None,
        epochs: int | None = None,
        init_milestones: Sequence[int] | None = None,
        milestones: Sequence[int] | None = None,
        init_lr_decay: float | None = None,
        lr_decay: float | None = None,
        lrate_decay: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(freeze_backbone=True, **kwargs)
        self.total_sessions = int(total_sessions)
        self.bal_epoch = int(bal_epoch)
        self.selector_epoch = int(selector_epoch)
        self.selector_lr = float(selector_lr)
        self.num_sampled_pcls = int(num_sampled_pcls)
        self.use_sm = bool(use_sm)
        self.margin_sample_num = int(margin_sample_num)
        self.use_oas_covariance = bool(use_oas_covariance)
        self.feature_scaling_mode = int(feature_scaling_mode)
        self.init_lr = None if init_lr is None else float(init_lr)
        self.lr = None if lr is None else float(lr)
        self.lrate = None if lrate is None else float(lrate)
        self.init_weight_decay = None if init_weight_decay is None else float(init_weight_decay)
        self.weight_decay = None if weight_decay is None else float(weight_decay)
        self.optimizer_type = str(optimizer_type)
        self.init_epoch = None if init_epoch is None else int(init_epoch)
        self.epochs = None if epochs is None else int(epochs)
        self.init_milestones = _as_int_list(init_milestones)
        self.milestones = _as_int_list(milestones)
        self.init_lr_decay = None if init_lr_decay is None else float(init_lr_decay)
        self.lr_decay = None if lr_decay is None else float(lr_decay)
        self.lrate_decay = None if lrate_decay is None else float(lrate_decay)
        self.naive = nn.ModuleDict()
        self.balanced = nn.ModuleDict()
        self.reverse = nn.ModuleDict()
        self.selector = DCESelector(int(self.detector.feature_dim), 3 * self.total_sessions)
        self.stats: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.current_key = "task0"

    def _ensure_experts(self, key: str) -> None:
        if key in self.naive:
            return
        dim = int(self.detector.feature_dim)
        self.naive[key] = DCEExpert(dim, self.num_classes)
        self.balanced[key] = DCEExpert(dim, self.num_classes)
        self.reverse[key] = DCEExpert(dim, self.num_classes)

    def _freeze_except_current(self) -> None:
        for pool in [self.naive, self.balanced, self.reverse]:
            for key, module in pool.items():
                for p in module.parameters():
                    p.requires_grad_(key == self.current_key)
        for p in self.selector.parameters():
            p.requires_grad_(False)

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.current_task_id = _task_id(task)
        self.current_key = f"task{self.current_task_id}"
        self._ensure_experts(self.current_key)
        self._freeze_except_current()

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        cfg = _official_optimizer_cfg(
            optimizer_cfg,
            self.current_task_id or 0,
            init_lr=self.init_lr,
            lr=self.lr,
            lrate=self.lrate,
            init_weight_decay=self.init_weight_decay,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
        )
        cfg.setdefault("lr", 0.01)
        cfg.setdefault("weight_decay", 5e-4)
        return build_optimizer(self.parameters(), cfg)

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        task_id = _task_id(task)
        epochs = _official_task_epochs(trainer, task_id, self.init_epoch, self.epochs)
        optimizer = self.configure_optimizer(trainer.optimizer_cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        _run_minibatch_loop(self, trainer, task, train_loader, optimizer, epochs, scheduler)
        return True

    def _counts(self, y: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(y.long(), minlength=self.num_classes).float().to(y.device)
        return counts.clamp_min(1.0)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        z = self.extract_features(batch["x"])
        y = _local_targets(batch["y"], self.num_classes)
        counts = self._counts(y)
        logits_n = self.naive[self.current_key](z)
        logits_b = self.balanced[self.current_key](z)
        logits_r = self.reverse[self.current_key](z)
        log_counts = counts.log().view(1, -1)
        loss_n = F.cross_entropy(logits_n, y)
        loss_b = F.cross_entropy(logits_b + log_counts, y)
        loss_r = F.cross_entropy(logits_r + 2.0 * log_counts, y)
        loss = loss_n + loss_b + loss_r
        return {"loss": loss, "ce": loss_n.detach(), "balanced_ce": loss_b.detach(), "reverse_ce": loss_r.detach()}

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is None:
            return
        features, labels = self.collect_features(train_loader)
        self._update_stats(self.current_key, features, labels)
        self._train_selector_from_stats()
        self._freeze_except_current()

    def _update_stats(self, key: str, features: torch.Tensor, labels: torch.Tensor) -> None:
        for cls in range(self.num_classes):
            z = features[labels == cls].float()
            if z.numel() == 0:
                continue
            mean = z.mean(dim=0)
            if self.use_oas_covariance and z.shape[0] >= 2:
                cov_np = OAS().fit(z.numpy()).covariance_
                cov = torch.as_tensor(cov_np, dtype=torch.float32)
            elif z.shape[0] >= max(self.margin_sample_num, 2):
                centered = z - mean
                cov = centered.t().matmul(centered) / max(z.shape[0] - 1, 1)
            else:
                cov = torch.eye(z.shape[1]) * 1e-3
            cov = cov + torch.eye(cov.shape[0]) * 1e-4
            self.stats[(key, cls)] = (mean.detach(), cov.detach(), int(z.shape[0]))

    def _sample_stats(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        xs = []
        ys = []
        cur_task = int(self.current_task_id or 0)
        for (key, cls), (mean, cov, _count) in self.stats.items():
            try:
                dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
                sample = dist.sample((self.num_sampled_pcls,))
            except Exception:
                sample = mean.unsqueeze(0).repeat(self.num_sampled_pcls, 1)
            task_id = int(key.replace("task", "")) if key.startswith("task") and key[4:].isdigit() else 0
            if self.feature_scaling_mode:
                rand_scaling = 0.02 * (torch.rand(sample.shape[0]) - 0.5)
                if self.feature_scaling_mode == 1:
                    factor = 1.0 + rand_scaling * max(cur_task - task_id, 0)
                elif self.feature_scaling_mode == 2:
                    factor = 1.0 + rand_scaling * max(cur_task, 0) / 2.0
                elif self.feature_scaling_mode == 4:
                    factor = 1.0 + rand_scaling * max(cur_task, 0)
                else:
                    factor = 1.0 + rand_scaling
                sample = sample / factor.clamp_min(1e-6).unsqueeze(1)
            xs.append(sample)
            ys.append(torch.full((sample.shape[0],), int(cls), dtype=torch.long))
        if not xs:
            return None
        return torch.cat(xs, dim=0).to(self.device), torch.cat(ys, dim=0).to(self.device)

    def _train_selector_from_stats(self) -> None:
        payload = self._sample_stats()
        if payload is None:
            return
        x, y = payload
        keys = list(self.naive.keys())
        if not keys:
            return
        for p in self.selector.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(self.selector.parameters(), lr=self.selector_lr, momentum=0.9, weight_decay=self.weight_decay or 2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.bal_epoch, self.selector_epoch, 1))
        self.selector.train()
        for _epoch in range(max(self.bal_epoch, self.selector_epoch, 1)):
            expert_logits = self._all_expert_logits(x, keys).detach()
            weights = self.selector(x)[:, : 3 * len(keys)]
            weights = F.softmax(weights, dim=1) if self.use_sm else weights
            logits = torch.einsum("be,bec->bc", weights, expert_logits)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        for p in self.selector.parameters():
            p.requires_grad_(False)

    def _all_expert_logits(self, z: torch.Tensor, keys: list[str]) -> torch.Tensor:
        logits = []
        for key in keys:
            logits.extend([self.naive[key](z), self.balanced[key](z), self.reverse[key](z)])
        return torch.stack(logits, dim=1)

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.extract_features(x)
        keys = list(self.naive.keys())
        expert_logits = self._all_expert_logits(z, keys)
        weights = self.selector(z)[:, : 3 * len(keys)]
        weights = F.softmax(weights, dim=1) if self.use_sm else weights
        logits = torch.einsum("be,bec->bc", weights, expert_logits)
        return {"logits": logits, "features": z, "expert_weights": weights.detach()}
