from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..losses import kd_loss
from ..memory import ReplayBuffer, extract_feature_table, sparse_uniform_indices
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, merge_batches


@register_method("sur_lid")
@register_method("sur-lid")
class SURLIDMethod(ContinualMethod):
    """Sparse Uniform Replay + Latent-space Incremental Detector.

    SUR stores stable, uniformly spread exemplars. LID isolates task-specific
    forgery latents and aligns historical replay features to their stored task
    centroids while training the shared binary detector.
    """

    def __init__(
        self,
        memory_size: int = 1500,
        memory_batch_size: int = 32,
        kd_weight: float = 1.0,
        isolation_weight: float = 0.1,
        alignment_weight: float = 0.1,
        margin: float = 0.5,
        temperature: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.memory = ReplayBuffer(memory_size, balanced=True, group_key="label")
        self.memory_batch_size = int(memory_batch_size)
        self.kd_weight = float(kd_weight)
        self.isolation_weight = float(isolation_weight)
        self.alignment_weight = float(alignment_weight)
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.teacher: torch.nn.Module | None = None
        self.task_fake_centroids: dict[int, torch.Tensor] = {}
        self.task_real_centroids: dict[int, torch.Tensor] = {}

    def _lid_losses(self, features: torch.Tensor, labels: torch.Tensor, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(features, dim=-1)
        iso = z.new_tensor(0.0)
        align = z.new_tensor(0.0)
        count_iso = 0
        count_align = 0
        cur_tasks = task_ids.unique(sorted=True).tolist()
        for tid in cur_tasks:
            tid = int(tid)
            fake_mask = (task_ids == tid) & (labels == 1)
            real_mask = (task_ids == tid) & (labels == 0)
            if fake_mask.sum() > 1 and self.task_fake_centroids:
                fake_c = F.normalize(z[fake_mask].mean(dim=0), dim=0)
                for old_tid, old_c in self.task_fake_centroids.items():
                    if old_tid == tid:
                        continue
                    old_c = F.normalize(old_c.to(z.device), dim=0)
                    iso = iso + F.relu(self.margin + torch.dot(fake_c, old_c))
                    count_iso += 1
            if fake_mask.sum() > 0 and tid in self.task_fake_centroids:
                c = F.normalize(self.task_fake_centroids[tid].to(z.device), dim=0)
                align = align + (1 - (z[fake_mask] @ c).mean())
                count_align += 1
            if real_mask.sum() > 0 and tid in self.task_real_centroids:
                c = F.normalize(self.task_real_centroids[tid].to(z.device), dim=0)
                align = align + (1 - (z[real_mask] @ c).mean())
                count_align += 1
        return iso / max(count_iso, 1), align / max(count_align, 1)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        mem = self.memory.sample(self.memory_batch_size, device=self.device)
        train_batch = merge_batches(batch, mem)
        out = self.predict(train_batch)
        y = train_batch["y"].long()
        ce = F.cross_entropy(out["logits"], y)
        task_ids = train_batch.get("task_id")
        if not torch.is_tensor(task_ids):
            task_ids = torch.full_like(y, int(getattr(task, "task_id", self.current_task_id or 0)))
        else:
            task_ids = task_ids.long().to(self.device)
        iso, align = self._lid_losses(out["features"], y, task_ids)
        loss = ce + self.isolation_weight * iso + self.alignment_weight * align
        log: dict[str, torch.Tensor] = {"ce": ce.detach(), "isolation": iso.detach(), "alignment": align.detach()}
        if self.teacher is not None and mem:
            with torch.no_grad():
                t_out = self.teacher(train_batch["x"])
            kd = kd_loss(out["logits"], t_out["logits"], self.temperature)
            loss = loss + self.kd_weight * kd
            log["kd"] = kd.detach()
        log["loss"] = loss
        return log

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is not None:
            feat, labels, rows = extract_feature_table(self, train_loader, self.device)
            tid = int(getattr(task, "task_id", self.current_task_id or 0))
            if feat.numel() > 0:
                z = F.normalize(feat.float(), dim=-1)
                fake = labels == 1
                real = labels == 0
                if fake.any():
                    self.task_fake_centroids[tid] = z[fake].mean(dim=0).detach().cpu()
                if real.any():
                    self.task_real_centroids[tid] = z[real].mean(dim=0).detach().cpu()
                if self.memory.capacity > 0:
                    stability = torch.linalg.norm(feat.float(), dim=1)
                    quota = max(self.memory.capacity // max(tid + 1, 1), 1)
                    idx = sparse_uniform_indices(feat, min(quota, len(rows)), stability=stability)
                    self.memory.replace(list(self.memory.samples) + [rows[i] for i in idx])
        self.teacher = self.frozen_detector_copy()
