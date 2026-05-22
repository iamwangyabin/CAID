from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..losses import feature_distillation_loss, kd_loss, supervised_contrastive_loss
from ..memory import ReplayBuffer, central_and_hard_indices, extract_feature_table
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, merge_batches


@register_method("dfil")
class DFILMethod(ContinualMethod):
    """Deepfake Incremental Learning with invariant forgery clues.

    Faithful implementation hooks:
      * supervised contrastive representation term,
      * feature-level and label/logit-level KD from historical model,
      * central + hard exemplar replay memory.
    """

    def __init__(
        self,
        memory_size: int = 1500,
        memory_batch_size: int = 32,
        supcon_weight: float = 0.1,
        kd_weight: float = 1.0,
        feature_kd_weight: float = 1.0,
        temperature: float = 2.0,
        hard_ratio: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.memory = ReplayBuffer(memory_size, balanced=True, group_key="label")
        self.memory_batch_size = int(memory_batch_size)
        self.supcon_weight = float(supcon_weight)
        self.kd_weight = float(kd_weight)
        self.feature_kd_weight = float(feature_kd_weight)
        self.temperature = float(temperature)
        self.hard_ratio = float(hard_ratio)
        self.teacher: torch.nn.Module | None = None

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        mem = self.memory.sample(self.memory_batch_size, device=self.device)
        train_batch = merge_batches(batch, mem)
        out = self.predict(train_batch)
        y = train_batch["y"].long()
        ce = F.cross_entropy(out["logits"], y)
        sup = supervised_contrastive_loss(out["features"], y)
        loss = ce + self.supcon_weight * sup
        log: dict[str, torch.Tensor] = {"ce": ce.detach(), "supcon": sup.detach()}
        if self.teacher is not None and mem:
            with torch.no_grad():
                t_out = self.teacher(train_batch["x"])
            kd = kd_loss(out["logits"], t_out["logits"], self.temperature)
            fkd = feature_distillation_loss(out["features"], t_out["features"])
            loss = loss + self.kd_weight * kd + self.feature_kd_weight * fkd
            log.update({"kd": kd.detach(), "feature_kd": fkd.detach()})
        log["loss"] = loss
        return log

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is not None and self.memory.capacity > 0:
            feat, labels, rows = extract_feature_table(self, train_loader, self.device)
            quota = max(self.memory.capacity // max((int(getattr(task, "task_id", 0)) + 1), 1), 1)
            idx = central_and_hard_indices(feat, labels, min(quota, len(rows)), hard_ratio=self.hard_ratio)
            new_rows = [rows[i] for i in idx]
            old_rows = list(self.memory.samples)
            self.memory.replace(old_rows + new_rows)
        self.teacher = self.frozen_detector_copy()
