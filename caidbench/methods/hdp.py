from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..losses import kd_loss
from ..registry import register_method
from .base import ContinualMethod, batch_to_device


@register_method("hdp")
class HDPMethod(ContinualMethod):
    """Historical Distribution Preserving continual face forgery detector.

    Maintains a universal adversarial perturbation (UAP) as a compact proxy for
    historical forgery distribution and distills historical real/fake responses
    from a frozen teacher.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        uap_shape: list[int] | tuple[int, ...] | None = None,
        kd_weight: float = 1.0,
        real_kd_weight: float = 1.0,
        adv_kd_weight: float = 1.0,
        temperature: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.kd_weight = float(kd_weight)
        self.real_kd_weight = float(real_kd_weight)
        self.adv_kd_weight = float(adv_kd_weight)
        self.temperature = float(temperature)
        self.teacher: nn.Module | None = None
        self.uap: nn.Parameter | None = None
        if uap_shape is not None:
            self._init_uap(tuple(int(x) for x in uap_shape))

    def _init_uap(self, sample_shape: tuple[int, ...]) -> None:
        shape = (1, *sample_shape) if len(sample_shape) >= 1 else (1,)
        self.uap = nn.Parameter(torch.zeros(shape))

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        super().before_task(task, train_loader)
        if self.uap is None and train_loader is not None:
            try:
                first = next(iter(train_loader))
                self._init_uap(tuple(first["x"].shape[1:]))
            except StopIteration:
                pass
        if self.uap is not None:
            self.uap.data = self.uap.data.to(self.device)

    def _add_uap(self, x: torch.Tensor) -> torch.Tensor:
        if self.uap is None:
            return x
        u = self.uap.to(x.device)
        return x + u

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        x = batch["x"]
        y = batch["y"].long()
        out = self.detector(x)
        ce = F.cross_entropy(out["logits"], y)
        loss = ce
        log: dict[str, torch.Tensor] = {"ce": ce.detach()}
        if self.teacher is not None and self.uap is not None:
            x_adv = self._add_uap(x)
            adv_out = self.detector(x_adv)
            with torch.no_grad():
                t_real = self.teacher(x)
                t_adv = self.teacher(x_adv)
            adv_kd = kd_loss(adv_out["logits"], t_adv["logits"], self.temperature)
            all_kd = kd_loss(out["logits"], t_real["logits"], self.temperature)
            real_mask = y == 0
            if real_mask.any():
                real_kd = kd_loss(out["logits"][real_mask], t_real["logits"][real_mask], self.temperature)
            else:
                real_kd = loss.new_tensor(0.0)
            loss = loss + self.kd_weight * all_kd + self.adv_kd_weight * adv_kd + self.real_kd_weight * real_kd
            log.update({"kd": all_kd.detach(), "adv_kd": adv_kd.detach(), "real_kd": real_kd.detach()})
        log["loss"] = loss
        return log

    def after_optimizer_step(self, task: Any | None = None) -> None:
        if self.uap is not None:
            self.uap.data.clamp_(-self.epsilon, self.epsilon)

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.teacher = self.frozen_detector_copy()
