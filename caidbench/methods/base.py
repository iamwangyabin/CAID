from __future__ import annotations

from abc import ABC
import copy
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from ..memory.replay import ReplayBuffer
from ..models.heads import build_detector


def batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def merge_batches(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if not a:
        return b
    if not b:
        return a
    out: dict[str, Any] = {}
    for k in set(a) | set(b):
        if k not in a:
            out[k] = b[k]
        elif k not in b:
            out[k] = a[k]
        elif torch.is_tensor(a[k]) and torch.is_tensor(b[k]):
            out[k] = torch.cat([a[k], b[k].to(a[k].device)], dim=0)
        elif isinstance(a[k], list) and isinstance(b[k], list):
            out[k] = a[k] + b[k]
        else:
            out[k] = a[k]
    return out


def build_optimizer(params: Iterable[nn.Parameter], cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
    cfg = cfg or {}
    trainable = [p for p in params if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters for optimizer")
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    name = str(cfg.get("type", "adamw")).lower()
    if name == "sgd":
        return torch.optim.SGD(trainable, lr=lr, weight_decay=weight_decay, momentum=float(cfg.get("momentum", 0.9)))
    if name == "adam":
        return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class ContinualMethod(nn.Module, ABC):
    """Base class for all methods in the benchmark.

    A method may override:
      - before_task: allocate task-specific prompts/adapters/experts
      - observe: one minibatch objective
      - transform_gradients: gradient projection / masking before optimizer.step
      - after_task: memory update, teacher update, importance estimation
      - fit_task: full custom loop. Return True when it handled training.
    """

    method_name = "base"

    def __init__(self, detector_cfg: dict[str, Any] | None = None, num_classes: int = 2, **kwargs: Any) -> None:
        super().__init__()
        detector_cfg = dict(detector_cfg or {})
        detector_cfg.setdefault("num_classes", num_classes)
        self.detector = build_detector(detector_cfg)
        self.num_classes = int(num_classes)
        self.current_task_id: int | None = None
        self.extra_cfg = dict(kwargs)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.current_task_id = int(getattr(task, "task_id", task if isinstance(task, int) else 0))

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        return build_optimizer(self.parameters(), optimizer_cfg)

    def frozen_detector_copy(self) -> nn.Module:
        return freeze_module(copy.deepcopy(self.detector).to(self.device))

    def auxiliary_state_dict(self) -> dict[str, Any]:
        return {
            name: value.state_dict()
            for name, value in self.__dict__.items()
            if isinstance(value, ReplayBuffer)
        }

    def load_auxiliary_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        for name, payload in state.items():
            value = getattr(self, name, None)
            if isinstance(value, ReplayBuffer) and isinstance(payload, dict):
                value.load_state_dict(payload)

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        return self.detector(x)

    def classification_loss(self, out: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out["logits"], y.long())

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        out = self.predict(batch)
        loss = self.classification_loss(out, batch["y"])
        return {"loss": loss, "ce": loss.detach(), "logits": out["logits"].detach()}

    def transform_gradients(self, task: Any | None = None) -> None:
        return None

    def after_optimizer_step(self, task: Any | None = None) -> None:
        return None

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        return None

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        return False
