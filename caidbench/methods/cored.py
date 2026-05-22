from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..losses import feature_distillation_loss, kd_loss
from ..registry import register_method
from .base import ContinualMethod, batch_to_device


@register_method("cored")
class CoReDMethod(ContinualMethod):
    """Continual Representation using Distillation.

    Implements the teacher-student structure used by CoReD: current-task CE plus
    logit and feature distillation from the previous frozen detector. The
    representation-learning component is exposed through the shared backbone and
    can be combined with replay through configs if desired.
    """

    def __init__(
        self,
        kd_weight: float = 1.0,
        feature_kd_weight: float = 1.0,
        temperature: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kd_weight = float(kd_weight)
        self.feature_kd_weight = float(feature_kd_weight)
        self.temperature = float(temperature)
        self.teacher: torch.nn.Module | None = None

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        out = self.predict(batch)
        y = batch["y"].long()
        ce = F.cross_entropy(out["logits"], y)
        loss = ce
        log: dict[str, torch.Tensor] = {"ce": ce.detach()}
        if self.teacher is not None:
            with torch.no_grad():
                t_out = self.teacher(batch["x"])
            kd = kd_loss(out["logits"], t_out["logits"], temperature=self.temperature)
            fkd = feature_distillation_loss(out["features"], t_out["features"])
            loss = loss + self.kd_weight * kd + self.feature_kd_weight * fkd
            log.update({"kd": kd.detach(), "feature_kd": fkd.detach()})
        log["loss"] = loss
        return log

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.teacher = self.frozen_detector_copy()
