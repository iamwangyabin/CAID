from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..losses import hsic_bottleneck_loss, kd_loss
from ..memory import ReplayBuffer, extract_feature_table, hsic_guided_indices
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, merge_batches


@register_method("hsic")
@register_method("hsic_bottleneck")
class HSICBottleneckMethod(ContinualMethod):
    """HSIC Bottleneck + HSIC-Guided Replay with online image features.

    The method no longer assumes offline CLIP feature extraction. In the
    default config, raw image tensors are passed through an online CLIP vision
    backbone inside ``self.detector`` before HSIC losses and HGR replay are
    computed.

    Nuisance variables default to generator/domain identifiers; task IDs are
    used when generator strings are absent.
    """

    def __init__(
        self,
        hsic_weight: float = 1.0,
        label_hsic_weight: float = 0.0,
        nuisance_hsic_weight: float = 1.0,
        hgr_lambda_kc: float = 0.5,
        memory_size: int = 1500,
        memory_batch_size: int = 32,
        kd_weight: float = 0.5,
        temperature: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hsic_weight = float(hsic_weight)
        self.label_hsic_weight = float(label_hsic_weight)
        self.nuisance_hsic_weight = float(nuisance_hsic_weight)
        self.hgr_lambda_kc = float(hgr_lambda_kc)
        self.memory = ReplayBuffer(memory_size, balanced=True, group_key="label")
        self.memory_batch_size = int(memory_batch_size)
        self.kd_weight = float(kd_weight)
        self.temperature = float(temperature)
        self.teacher: torch.nn.Module | None = None
        self._nuisance_to_id: dict[str, int] = {}

    def _nuisance_ids(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        gens = batch.get("generator") or batch.get("domain")
        if gens is not None and not torch.is_tensor(gens):
            ids = []
            for g in gens:
                key = str(g)
                if key not in self._nuisance_to_id:
                    self._nuisance_to_id[key] = len(self._nuisance_to_id)
                ids.append(self._nuisance_to_id[key])
            return torch.tensor(ids, dtype=torch.long, device=device)
        if torch.is_tensor(gens):
            return gens.long().to(device)
        return batch["task_id"].long().to(device)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        mem = self.memory.sample(self.memory_batch_size, device=self.device)
        train_batch = merge_batches(batch, mem)
        out = self.predict(train_batch)
        y = train_batch["y"].long()
        ce = F.cross_entropy(out["logits"], y)
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        nuisance = self._nuisance_ids(train_batch, self.device)
        hsic_loss = hsic_bottleneck_loss(
            out["features"],
            y_onehot,
            nuisances=[nuisance],
            lambda_label=self.label_hsic_weight,
            lambda_nuisance=self.nuisance_hsic_weight,
        )
        loss = ce + self.hsic_weight * hsic_loss
        log: dict[str, torch.Tensor] = {"ce": ce.detach(), "hsic": hsic_loss.detach()}
        if self.teacher is not None and mem:
            with torch.no_grad():
                t_out = self.teacher(train_batch["x"])
            kd = kd_loss(out["logits"], t_out["logits"], self.temperature)
            loss = loss + self.kd_weight * kd
            log["kd"] = kd.detach()
        log["loss"] = loss
        return log

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is not None and self.memory.capacity > 0:
            feat, labels, rows = extract_feature_table(self, train_loader, self.device)
            nuisance_rows = []
            for r in rows:
                key = str(r.get("generator", r.get("domain", r.get("task_id", "unknown"))))
                if key not in self._nuisance_to_id:
                    self._nuisance_to_id[key] = len(self._nuisance_to_id)
                nuisance_rows.append(self._nuisance_to_id[key])
            nuisance = torch.tensor(nuisance_rows, dtype=torch.long) if nuisance_rows else None
            quota = max(self.memory.capacity // max((int(getattr(task, "task_id", 0)) + 1), 1), 1)
            idx = hsic_guided_indices(feat, labels, nuisance, min(quota, len(rows)), lambda_kc=self.hgr_lambda_kc)
            self.memory.replace(list(self.memory.samples) + [rows[i] for i in idx])
        self.teacher = self.frozen_detector_copy()
