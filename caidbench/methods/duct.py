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
        raise ValueError("Labels must be non-negative for domain-incremental methods.")
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
        self.in_features = int(in_dim)
        self.out_features = int(out_dim)
        self.weight = nn.Parameter(torch.empty(int(out_dim), int(in_dim)))
        self.sigma = nn.Parameter(torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.sigma.data.fill_(1.0)

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
        use_official_retrain_lr: bool = True,
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
        self.use_official_retrain_lr = bool(use_official_retrain_lr)
        self.expanded_head: ExpandedCosineHead | None = None
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
        if self.expanded_head is not None:
            return int(self.expanded_head.out_features)
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

    def load_state_dict(self, state_dict: Any, strict: bool = True):  # type: ignore[override]
        head_weight = state_dict.get("expanded_head.weight") if hasattr(state_dict, "get") else None
        if torch.is_tensor(head_weight):
            self._update_fc(int(head_weight.shape[0]))
        return super().load_state_dict(state_dict, strict=strict)

    def _require_head(self) -> ExpandedCosineHead:
        if self.expanded_head is None:
            self._update_fc(max((self._current_index + 1) * self.increment, self.increment))
        assert self.expanded_head is not None
        return self.expanded_head

    def _update_fc(self, total_classes: int) -> None:
        total_classes = int(total_classes)
        old_weight = None
        if self.expanded_head is not None:
            if int(self.expanded_head.out_features) == total_classes:
                return
            old_weight = self.expanded_head.weight.detach().clone()
        head = ExpandedCosineHead(int(self.detector.feature_dim), total_classes).to(self.device)
        if old_weight is not None:
            rows = min(old_weight.shape[0], head.weight.shape[0])
            head.weight.data[:rows] = old_weight[:rows].to(device=head.weight.device, dtype=head.weight.dtype)
        self.expanded_head = head

    def _make_optimizer(self, trainer: Any) -> torch.optim.Optimizer:
        base_lr = float(self.lrate if self.lrate is not None else trainer.optimizer_cfg.get("lr", 0.1))
        backbone_params = [p for p in self.detector.backbone.parameters() if p.requires_grad]
        head = self._require_head()
        head_params = [head.weight, head.sigma]
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
        self._update_fc(max((self._current_index + 1) * self.increment, self.increment))
        if self.reset_backbone_each_task:
            _load_float_state(self.detector.backbone, self._init_backbone_state)
        self._compute_current_class_means(trainer, train_loader)
        self.train()
        for p in self.detector.head.parameters():
            p.requires_grad_(False)
        optimizer = self._make_optimizer(trainer)
        scheduler = self._make_scheduler(optimizer, trainer)
        for epoch in range(trainer.max_epochs):
            epoch_lr = float(optimizer.param_groups[0]["lr"])
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
                trainer.logger.info(
                    "task=%s epoch=%d/%d %s",
                    task.name,
                    epoch + 1,
                    trainer.max_epochs,
                    ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
                )
                trainer.log_metrics(
                    {
                        **{f"train/{key}": value for key, value in metrics.items()},
                        "train/task_index": float(_task_id(task)),
                        "train/epoch": epoch + 1,
                        "train/lr": epoch_lr,
                    }
                )
        self._merge_backbone()
        self._retrain_head(trainer, train_loader)
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
        head = self._require_head()
        for p in self.detector.backbone.parameters():
            p.requires_grad_(False)
        head.weight.requires_grad_(True)
        head.sigma.requires_grad_(True)
        base_lr = float(self.lrate if self.lrate is not None else trainer.optimizer_cfg.get("lr", self.lr_re))
        retrain_lr = base_lr if self.use_official_retrain_lr else self.lr_re
        trainer.logger.info("DUCT retrain head epochs=%d lr=%.6g", self.retrain_epochs, retrain_lr)
        optimizer = torch.optim.SGD([head.weight, head.sigma], lr=retrain_lr, momentum=0.9, weight_decay=self.weight_decay)
        self.train()
        active = self._current_slice()
        for _epoch in range(self.retrain_epochs):
            for batch in train_loader:
                batch = batch_to_device(batch, self.device)
                z = self.detector.extract_features(batch["x"])
                logits = head(z)[:, active]
                loss = F.cross_entropy(logits, _local_targets(batch["y"], self.increment))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                trainer.advance_step()
        for p in self.detector.backbone.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def _compute_current_class_means(self, trainer: Any, train_loader: Any) -> None:
        try:
            mean_loader = trainer.dataloader(self._current_index, "train", shuffle=False, transform_split="test", drop_last=False)
        except TypeError:
            mean_loader = train_loader
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        was_training = self.training
        self.eval()
        for batch in mean_loader:
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
        head = self._require_head()
        cur_start = self._current_index * self.increment
        cur_end = cur_start + self.increment
        current_classes = list(range(cur_start, cur_end))
        if not all(cls in self._class_means for cls in current_classes):
            return
        cur_means = torch.stack([self._class_means[cls] for cls in current_classes]).to(self.device)
        cur_weight = head.weight[cur_start:cur_end].detach()
        for old_index in range(self._current_index):
            old_start = old_index * self.increment
            old_end = old_start + self.increment
            old_classes = list(range(old_start, old_end))
            if not all(cls in self._class_means for cls in old_classes):
                continue
            old_means = torch.stack([self._class_means[cls] for cls in old_classes]).to(self.device)
            cost = torch.cdist(cur_means.float(), old_means.float(), p=2)
            transport = _sinkhorn_uniform(cost, self.ot_reg).to(device=self.device)
            transported = transport.t().matmul(
                cur_weight.to(device=self.device, dtype=transport.dtype)
            ).to(dtype=head.weight.dtype)
            old_weight = head.weight[old_start:old_end]
            old_weight.copy_((1.0 - self.head_merge_ratio) * old_weight + self.head_merge_ratio * transported)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        z = self.detector.extract_features(batch["x"])
        logits = self._require_head()(z)[:, self._current_slice()]
        ce = F.cross_entropy(logits, _local_targets(batch["y"], self.increment))
        return {"loss": ce, "ce": ce.detach()}

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.detector.extract_features(x)
        expanded = self._require_head()(z)[:, : self._seen_classes]
        folded = []
        for cls in range(self.num_classes):
            folded.append(expanded[:, cls :: self.increment].max(dim=1).values)
        logits = torch.stack(folded, dim=1)
        return {"logits": logits, "features": z, "expanded_logits": expanded}
