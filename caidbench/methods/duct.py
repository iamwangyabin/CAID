from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..registry import register_method
from .base import ContinualMethod, batch_to_device


def _local_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = y.long()
    if y.numel() and (int(y.min()) < 0):
        raise ValueError("Labels must be non-negative for official DIL methods.")
    return torch.remainder(y, int(num_classes))


def _task_id(task: Any) -> int:
    return int(getattr(task, "task_id", task if isinstance(task, int) else 0))


def _as_int_list(value: Sequence[int] | None) -> list[int]:
    return [int(v) for v in value] if value is not None else []


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items() if torch.is_floating_point(v)}


def _state_delta(current: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in current.items():
        if key in base and value.shape == base[key].shape:
            out[key] = value.detach().cpu() - base[key]
    return out


def _load_float_state(module: nn.Module, float_state: dict[str, torch.Tensor]) -> None:
    state = module.state_dict()
    for key, value in float_state.items():
        if key in state and state[key].shape == value.shape:
            state[key] = value.to(device=state[key].device, dtype=state[key].dtype)
    module.load_state_dict(state, strict=False)


class ExpandedCosineHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(out_dim), int(in_dim)))
        self.sigma = nn.Parameter(torch.ones(1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(F.normalize(x.float(), dim=-1), F.normalize(self.weight, dim=-1))
        return logits * self.sigma


def _sinkhorn_uniform(cost: torch.Tensor, reg: float, iters: int = 100) -> torch.Tensor:
    reg = max(float(reg), 1e-6)
    cost = cost.double()
    n, m = cost.shape
    a = torch.full((n,), 1.0 / max(n, 1), dtype=torch.double, device=cost.device)
    b = torch.full((m,), 1.0 / max(m, 1), dtype=torch.double, device=cost.device)
    kernel = torch.exp(-cost / reg).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(int(iters)):
        u = a / (kernel @ v).clamp_min(1e-12)
        v = b / (kernel.t() @ u).clamp_min(1e-12)
    return torch.diag(u) @ kernel @ torch.diag(v)


@register_method("duct")
class DUCTMethod(ContinualMethod):
    """DUCT representation consolidation with domain-expanded cosine heads."""

    def __init__(
        self,
        merge_scalar: float = 0.5,
        retrain_epochs: int = 5,
        epc_re: int | None = None,
        lr_re: float = 1e-3,
        head_merge_ratio: float = 0.5,
        bcb_lr_scale: float = 0.01,
        ot_reg: float = 0.1,
        lrate: float | None = None,
        weight_decay: float = 5e-4,
        milestones: Sequence[int] | None = None,
        lrate_decay: float | None = None,
        increment: int | None = None,
        total_sessions: int = 5,
        reset_backbone_each_task: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.merge_scalar = float(merge_scalar)
        self.retrain_epochs = int(retrain_epochs if epc_re is None else epc_re)
        self.lr_re = float(lr_re)
        self.head_merge_ratio = float(head_merge_ratio)
        self.bcb_lr_scale = float(bcb_lr_scale)
        self.ot_reg = float(ot_reg)
        self.lrate = None if lrate is None else float(lrate)
        self.weight_decay = float(weight_decay)
        self.milestones = _as_int_list(milestones)
        self.lrate_decay = None if lrate_decay is None else float(lrate_decay)
        self.increment = int(increment or self.num_classes)
        self.total_sessions = int(total_sessions)
        self.reset_backbone_each_task = bool(reset_backbone_each_task)
        self.max_expanded_classes = max(self.increment * self.total_sessions, self.num_classes)
        self.expanded_head = ExpandedCosineHead(int(self.detector.feature_dim), self.max_expanded_classes)
        for p in self.detector.head.parameters():
            p.requires_grad_(False)
        self._task_keys: list[int] = []
        self._current_index = 0
        self._init_backbone_state = _clone_state(self.detector.backbone)
        self._merged_delta: dict[str, torch.Tensor] = {
            key: torch.zeros_like(value) for key, value in self._init_backbone_state.items()
        }
        self._class_means: dict[int, torch.Tensor] = {}

    @property
    def _seen_classes(self) -> int:
        return max((len(self._task_keys) or 1) * self.increment, self.increment)

    def _current_slice(self) -> slice:
        start = self._current_index * self.increment
        return slice(start, start + self.increment)

    def _current_global_labels(self, y: torch.Tensor) -> torch.Tensor:
        return _local_targets(y, self.increment) + self._current_index * self.increment

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        super().before_task(task, train_loader)
        tid = _task_id(task)
        if tid not in self._task_keys:
            self._task_keys.append(tid)
        self._current_index = self._task_keys.index(tid)

    def _make_optimizer(self, trainer: Any) -> torch.optim.Optimizer:
        base_lr = float(self.lrate if self.lrate is not None else trainer.optimizer_cfg.get("lr", 0.1))
        backbone_params = [p for p in self.detector.backbone.parameters() if p.requires_grad]
        head_params = [self.expanded_head.weight, self.expanded_head.sigma]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * self.bcb_lr_scale, "weight_decay": self.weight_decay})
        groups.append({"params": head_params, "lr": base_lr, "weight_decay": self.weight_decay})
        return torch.optim.SGD(groups, momentum=0.9)

    def _make_scheduler(self, optimizer: torch.optim.Optimizer, trainer: Any) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.milestones and self.lrate_decay is not None:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=float(self.lrate_decay))
        return None

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        if self.reset_backbone_each_task:
            _load_float_state(self.detector.backbone, self._init_backbone_state)
        self.train()
        for p in self.detector.head.parameters():
            p.requires_grad_(False)
        optimizer = self._make_optimizer(trainer)
        scheduler = self._make_scheduler(optimizer, trainer)
        for epoch in range(trainer.max_epochs):
            totals: dict[str, float] = {}
            n = 0
            for batch in train_loader:
                out = self.observe(batch, task)
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                totals["ce"] = totals.get("ce", 0.0) + float(out["ce"].detach().cpu())
                n += 1
            if scheduler is not None:
                scheduler.step()
            if totals:
                trainer.logger.info("task=%s epoch=%d/%d ce=%.4f", task.name, epoch + 1, trainer.max_epochs, totals["ce"] / max(n, 1))
        self._merge_backbone()
        self._retrain_head(trainer, train_loader)
        self._store_class_means(train_loader)
        self._transport_classifier()
        return True

    def _merge_backbone(self) -> None:
        current = _clone_state(self.detector.backbone)
        delta = _state_delta(current, self._init_backbone_state)
        for key, value in delta.items():
            self._merged_delta[key] = self._merged_delta[key] + self.merge_scalar * value
        merged = {
            key: self._init_backbone_state[key] + self._merged_delta.get(key, torch.zeros_like(self._init_backbone_state[key]))
            for key in self._init_backbone_state
        }
        _load_float_state(self.detector.backbone, merged)

    def _retrain_head(self, trainer: Any, train_loader: Any) -> None:
        if self.retrain_epochs <= 0:
            return
        for p in self.detector.backbone.parameters():
            p.requires_grad_(False)
        self.expanded_head.weight.requires_grad_(True)
        self.expanded_head.sigma.requires_grad_(True)
        optimizer = torch.optim.SGD([self.expanded_head.weight, self.expanded_head.sigma], lr=self.lr_re, momentum=0.9, weight_decay=self.weight_decay)
        self.train()
        active = self._current_slice()
        for _epoch in range(self.retrain_epochs):
            for batch in train_loader:
                batch = batch_to_device(batch, self.device)
                z = self.detector.extract_features(batch["x"])
                logits = self.expanded_head(z)[:, active]
                loss = F.cross_entropy(logits, _local_targets(batch["y"], self.increment))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                trainer.advance_step()
        for p in self.detector.backbone.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def _store_class_means(self, train_loader: Any) -> None:
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        was_training = self.training
        self.eval()
        for batch in train_loader:
            batch = batch_to_device(batch, self.device)
            features.append(self.detector.extract_features(batch["x"]).detach().cpu())
            labels.append(self._current_global_labels(batch["y"].detach().cpu()))
        if was_training:
            self.train()
        if not features:
            return
        z = torch.cat(features, dim=0)
        y = torch.cat(labels, dim=0)
        for cls in y.unique().tolist():
            cls_int = int(cls)
            self._class_means[cls_int] = z[y == cls_int].float().mean(dim=0)

    @torch.no_grad()
    def _transport_classifier(self) -> None:
        if self._current_index == 0:
            return
        cur_start = self._current_index * self.increment
        cur_end = cur_start + self.increment
        current_classes = list(range(cur_start, cur_end))
        if not all(cls in self._class_means for cls in current_classes):
            return
        cur_means = torch.stack([self._class_means[cls] for cls in current_classes]).to(self.device)
        cur_weight = self.expanded_head.weight[cur_start:cur_end].detach()
        for old_index in range(self._current_index):
            old_start = old_index * self.increment
            old_end = old_start + self.increment
            old_classes = list(range(old_start, old_end))
            if not all(cls in self._class_means for cls in old_classes):
                continue
            old_means = torch.stack([self._class_means[cls] for cls in old_classes]).to(self.device)
            cost = torch.cdist(cur_means.float(), old_means.float(), p=2)
            transport = _sinkhorn_uniform(cost, self.ot_reg).to(device=self.device, dtype=self.expanded_head.weight.dtype)
            transported = transport.t().matmul(cur_weight.to(torch.double)).to(dtype=self.expanded_head.weight.dtype)
            old_weight = self.expanded_head.weight[old_start:old_end]
            old_weight.copy_((1.0 - self.head_merge_ratio) * old_weight + self.head_merge_ratio * transported)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        z = self.detector.extract_features(batch["x"])
        logits = self.expanded_head(z)[:, self._current_slice()]
        ce = F.cross_entropy(logits, _local_targets(batch["y"], self.increment))
        return {"loss": ce, "ce": ce.detach()}

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.detector.extract_features(x)
        expanded = self.expanded_head(z)[:, : self._seen_classes]
        folded = []
        for cls in range(self.num_classes):
            folded.append(expanded[:, cls :: self.increment].max(dim=1).values)
        logits = torch.stack(folded, dim=1)
        return {"logits": logits, "features": z, "expanded_logits": expanded}
