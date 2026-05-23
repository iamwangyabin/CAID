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
        representation_kd_weight: float = 0.0,
        temperature: float = 2.0,
        confidence_bins: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kd_weight = float(kd_weight)
        self.feature_kd_weight = float(feature_kd_weight)
        self.representation_kd_weight = float(representation_kd_weight)
        self.temperature = float(temperature)
        self.confidence_bins = int(confidence_bins)
        self.teacher: torch.nn.Module | None = None
        self.representation_memory: dict[tuple[int, int], torch.Tensor] = {}

    @staticmethod
    def _flat_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim > 2:
            return F.adaptive_avg_pool2d(features.float(), (1, 1)).flatten(1)
        return features.float()

    def _confidence_bin(self, confidence: torch.Tensor) -> torch.Tensor:
        bins = torch.ceil((confidence.float() - 0.5).clamp_min(0.0) * 10.0).long() - 1
        return bins.clamp(0, max(self.confidence_bins - 1, 0))

    @torch.no_grad()
    def _build_representation_memory(self, train_loader: Any | None) -> None:
        self.representation_memory.clear()
        if self.teacher is None or train_loader is None or self.representation_kd_weight <= 0:
            return
        sums: dict[tuple[int, int], torch.Tensor] = {}
        counts: dict[tuple[int, int], int] = {}
        was_training = self.teacher.training
        self.teacher.eval()
        for batch in train_loader:
            batch = batch_to_device(batch, self.device)
            out = self.teacher(batch["x"])
            prob = F.softmax(out["logits"], dim=-1)
            conf, pred = prob.max(dim=-1)
            y = batch["y"].long()
            keep = pred.eq(y)
            if not keep.any():
                continue
            feat = self._flat_features(out["features"])
            bins = self._confidence_bin(conf)
            for cls in y.unique(sorted=True):
                for bin_id in bins.unique(sorted=True):
                    mask = keep & y.eq(cls) & bins.eq(bin_id)
                    if not mask.any():
                        continue
                    key = (int(cls.item()), int(bin_id.item()))
                    sums[key] = sums.get(key, torch.zeros_like(feat[mask][0].detach().cpu())) + feat[mask].sum(dim=0).detach().cpu()
                    counts[key] = counts.get(key, 0) + int(mask.sum().item())
        self.teacher.train(was_training)
        for key, value in sums.items():
            self.representation_memory[key] = value / max(counts.get(key, 1), 1)

    def _representation_memory_loss(self, student_features: torch.Tensor, y: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if not self.representation_memory or self.representation_kd_weight <= 0:
            return student_features.new_tensor(0.0)
        feat = self._flat_features(student_features)
        prob = F.softmax(teacher_logits.detach(), dim=-1)
        conf, pred = prob.max(dim=-1)
        bins = self._confidence_bin(conf)
        loss = feat.new_tensor(0.0)
        count = 0
        for key, target in self.representation_memory.items():
            cls, bin_id = key
            mask = y.eq(cls) & pred.eq(y) & bins.eq(bin_id)
            if not mask.any():
                continue
            target = target.to(device=feat.device, dtype=feat.dtype)
            loss = loss + F.mse_loss(feat[mask].mean(dim=0), target)
            count += 1
        return loss / max(count, 1)

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
            rep = self._representation_memory_loss(out["features"], y, t_out["logits"])
            loss = loss + self.kd_weight * kd + self.feature_kd_weight * fkd + self.representation_kd_weight * rep
            log.update({"kd": kd.detach(), "feature_kd": fkd.detach(), "representation_kd": rep.detach()})
        log["loss"] = loss
        return log

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        super().before_task(task, train_loader)
        self._build_representation_memory(train_loader)

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.teacher = self.frozen_detector_copy()
