from __future__ import annotations

from typing import Any

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


@register_method("duct")
class DUCTMethod(ContinualMethod):
    """DUCT representation consolidation with a framework-compatible head."""

    def __init__(
        self,
        merge_scalar: float = 0.5,
        retrain_epochs: int = 5,
        epc_re: int | None = None,
        lr_re: float = 1e-3,
        head_merge_ratio: float = 0.5,
        bcb_lr_scale: float = 0.01,
        ot_reg: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.merge_scalar = float(merge_scalar)
        self.retrain_epochs = int(retrain_epochs if epc_re is None else epc_re)
        self.lr_re = float(lr_re)
        self.head_merge_ratio = float(head_merge_ratio)
        self.bcb_lr_scale = float(bcb_lr_scale)
        self.ot_reg = float(ot_reg)
        self._init_backbone_state = _clone_state(self.detector.backbone)
        self._merged_delta: dict[str, torch.Tensor] = {
            key: torch.zeros_like(value) for key, value in self._init_backbone_state.items()
        }
        self._previous_head: dict[str, torch.Tensor] | None = None

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        self.train()
        optimizer = self.configure_optimizer(trainer.optimizer_cfg)
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
            if totals:
                trainer.logger.info("task=%s epoch=%d/%d ce=%.4f", task.name, epoch + 1, trainer.max_epochs, totals["ce"] / max(n, 1))
        self._merge_backbone()
        self._retrain_head(trainer, train_loader)
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
        for p in self.detector.head.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(self.detector.head.parameters(), lr=self.lr_re, momentum=0.9)
        self.train()
        for _epoch in range(self.retrain_epochs):
            for batch in train_loader:
                batch = batch_to_device(batch, self.device)
                out = self.detector(batch["x"])
                loss = F.cross_entropy(out["logits"], _local_targets(batch["y"], self.num_classes))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                trainer.advance_step()
        if self._previous_head is not None:
            state = self.detector.head.state_dict()
            for key, value in state.items():
                if key in self._previous_head and torch.is_floating_point(value) and value.shape == self._previous_head[key].shape:
                    state[key] = self.head_merge_ratio * self._previous_head[key].to(value.device) + (1.0 - self.head_merge_ratio) * value
            self.detector.head.load_state_dict(state, strict=False)
        self._previous_head = {k: v.detach().cpu().clone() for k, v in self.detector.head.state_dict().items() if torch.is_floating_point(v)}
        for p in self.detector.backbone.parameters():
            p.requires_grad_(True)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        out = self.detector(batch["x"])
        ce = F.cross_entropy(out["logits"], _local_targets(batch["y"], self.num_classes))
        return {"loss": ce, "ce": ce.detach()}
