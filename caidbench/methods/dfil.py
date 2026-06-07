from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..losses import feature_distillation_loss, kd_loss, supervised_contrastive_loss
from ..memory import ReplayBuffer, central_and_hard_indices, dfil_official_indices, extract_feature_logit_table
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
        kd_alpha: float | None = None,
        feature_kd_weight: float = 1.0,
        temperature: float = 2.0,
        hard_ratio: float = 0.5,
        center_samples_per_class: int | None = None,
        hard_samples_per_class: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.memory = ReplayBuffer(memory_size, balanced=True, group_key="label")
        self.memory_batch_size = int(memory_batch_size)
        self.supcon_weight = float(supcon_weight)
        self.kd_weight = float(kd_weight if kd_alpha is None else kd_alpha)
        self.feature_kd_weight = float(feature_kd_weight)
        self.temperature = float(temperature)
        self.hard_ratio = float(hard_ratio)
        self.center_samples_per_class = None if center_samples_per_class is None else int(center_samples_per_class)
        self.hard_samples_per_class = None if hard_samples_per_class is None else int(hard_samples_per_class)
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

    @staticmethod
    def _configure_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler | None:
        name = train_cfg.get("lr_scheduler")
        if name is None or str(name).lower() in {"none", "null", ""}:
            return None
        if str(name).lower() != "step":
            raise NotImplementedError(f"DFIL official reproduction supports lr_scheduler=step, got {name!r}")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("step_size", 5)),
            gamma=float(train_cfg.get("gamma", 0.5)),
        )

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        train_cfg = getattr(trainer, "cfg", {}).get("train", {})
        if train_cfg.get("lr_scheduler") is None:
            return False
        self.train()
        optimizer = self.configure_optimizer(getattr(trainer, "optimizer_cfg", None))
        scheduler = self._configure_scheduler(optimizer, train_cfg)
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
                for key, value in self.train_metrics(out).items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                n += 1
            if scheduler is not None:
                scheduler.step()
            if totals:
                metrics = {k: v / max(n, 1) for k, v in totals.items()}
                trainer.log_train_metrics(metrics, task=task, epoch=epoch + 1, epochs=trainer.max_epochs, optimizer=optimizer)
        return True

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is not None and self.memory.capacity > 0:
            feat, labels, logits, rows = extract_feature_logit_table(self, train_loader, self.device)
            quota = max(self.memory.capacity // max((int(getattr(task, "task_id", 0)) + 1), 1), 1)
            if self.center_samples_per_class is not None or self.hard_samples_per_class is not None:
                center_n = self.center_samples_per_class if self.center_samples_per_class is not None else 0
                hard_n = self.hard_samples_per_class if self.hard_samples_per_class is not None else 0
                idx = dfil_official_indices(feat, labels, logits, center_per_class=center_n, hard_per_class=hard_n)
                idx = idx[: min(quota, len(rows))]
            else:
                idx = central_and_hard_indices(feat, labels, min(quota, len(rows)), hard_ratio=self.hard_ratio)
            new_rows = [rows[i] for i in idx]
            old_rows = list(self.memory.samples)
            self.memory.replace(old_rows + new_rows)
        self.teacher = self.frozen_detector_copy()
