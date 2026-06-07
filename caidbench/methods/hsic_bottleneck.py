from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..losses import hsic_bottleneck_loss, kd_loss, official_hsic_bottleneck_loss
from ..memory import ReplayBuffer, extract_feature_table, hsic_guided_indices, official_hgr_indices
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, merge_batches


class OnlineHSICBottleneckDetector(nn.Module):
    """Official-style HSIC model over online-computed backbone features."""

    def __init__(
        self,
        detector: nn.Module,
        bottleneck_dim: int = 64,
        num_classes: int = 1,
        freeze_backbone: bool = True,
        detach_input_features: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.input_dim = int(detector.feature_dim)
        self.feature_dim = int(bottleneck_dim)
        self.encoder = nn.Linear(self.input_dim, self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, int(num_classes))
        self.detach_input_features = bool(detach_input_features)
        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if all(not p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def extract_input_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.extract_input_features(x)
        enc_in = raw.detach() if self.detach_input_features else raw
        return self.encoder(enc_in)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.extract_input_features(x)
        enc_in = raw.detach() if self.detach_input_features else raw
        z = self.encoder(enc_in)
        logits = self.classifier(z)
        return {"logits": logits, "features": z, "input_features": raw.detach()}


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
        objective: str = "caid",
        lambda_x: float = 500.0,
        lambda_y: float = 300.0,
        buffer_lambda_x: float | None = None,
        buffer_lambda_y: float | None = None,
        bottleneck_dim: int = 64,
        binary_sigmoid: bool | None = None,
        hgr_alpha: float = 0.5,
        hgr_keep_frac: float | None = None,
        freeze_backbone_for_official: bool = True,
        detach_input_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.objective = str(objective).lower()
        self.official_equivalent = self.objective in {"official", "official_equivalent", "dual_hsic"}
        self.hsic_weight = float(hsic_weight)
        self.label_hsic_weight = float(label_hsic_weight)
        self.nuisance_hsic_weight = float(nuisance_hsic_weight)
        self.hgr_lambda_kc = float(hgr_lambda_kc)
        self.hgr_alpha = float(hgr_alpha)
        self.hgr_keep_frac = None if hgr_keep_frac is None else float(hgr_keep_frac)
        self.memory = ReplayBuffer(memory_size, balanced=True, group_key="label")
        self.memory_batch_size = int(memory_batch_size)
        self.kd_weight = float(kd_weight)
        self.temperature = float(temperature)
        self.lambda_x = float(lambda_x)
        self.lambda_y = float(lambda_y)
        self.buffer_lambda_x = float(self.lambda_x if buffer_lambda_x is None else buffer_lambda_x)
        self.buffer_lambda_y = float(self.lambda_y if buffer_lambda_y is None else buffer_lambda_y)
        self.binary_sigmoid = self.official_equivalent if binary_sigmoid is None else bool(binary_sigmoid)
        if self.official_equivalent:
            self.detector = OnlineHSICBottleneckDetector(
                self.detector,
                bottleneck_dim=int(bottleneck_dim),
                num_classes=1 if self.binary_sigmoid else self.num_classes,
                freeze_backbone=bool(freeze_backbone_for_official),
                detach_input_features=bool(detach_input_features),
            )
        self.teacher: torch.nn.Module | None = None
        self._nuisance_to_id: dict[str, int] = {}

    def _nuisance_ids(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        gens = batch["generator"] if "generator" in batch else batch.get("domain")
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
        task_id = batch.get("task_id")
        if torch.is_tensor(task_id):
            return task_id.long().to(device)
        size_source = batch.get("y", batch.get("x"))
        batch_size = int(size_source.shape[0]) if torch.is_tensor(size_source) and size_source.ndim > 0 else 1
        fallback = 0 if self.current_task_id is None else int(self.current_task_id)
        return torch.full((batch_size,), fallback, dtype=torch.long, device=device)

    def _classification_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.binary_sigmoid or logits.ndim == 1 or logits.shape[-1] == 1:
            return F.binary_cross_entropy_with_logits(logits.reshape(-1), y.float())
        return F.cross_entropy(logits, y.long())

    def _official_loss(self, batch: dict[str, Any], lambda_x: float, lambda_y: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self.predict(batch)
        y = batch["y"].long()
        ce = self._classification_loss(out["logits"], y)
        hsic_loss = official_hsic_bottleneck_loss(out["features"], out["input_features"], y, lambda_x=lambda_x, lambda_y=lambda_y)
        loss = ce + hsic_loss
        return loss, {"ce": ce.detach(), "hsic": hsic_loss.detach(), "logits": out["logits"].detach()}

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        if self.official_equivalent:
            loss, log = self._official_loss(batch, self.lambda_x, self.lambda_y)
            mem = self.memory.sample(self.memory_batch_size, device=self.device)
            if mem:
                mem_loss, mem_log = self._official_loss(mem, self.buffer_lambda_x, self.buffer_lambda_y)
                loss = loss + mem_loss
                log["replay_ce"] = mem_log["ce"]
                log["replay_hsic"] = mem_log["hsic"]
            log["loss"] = loss
            return log

        mem = self.memory.sample(self.memory_batch_size, device=self.device)
        train_batch = merge_batches(batch, mem)
        out = self.predict(train_batch)
        y = train_batch["y"].long()
        ce = self._classification_loss(out["logits"], y)
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
            if self.hgr_keep_frac is not None and self.hgr_keep_frac > 0:
                quota = max(1, math.ceil(len(rows) * self.hgr_keep_frac))
            else:
                quota = max(self.memory.capacity // max((int(getattr(task, "task_id", 0)) + 1), 1), 1)
            quota = min(quota, len(rows))
            if self.official_equivalent:
                idx = official_hgr_indices(feat, labels, quota, alpha=self.hgr_alpha)
            else:
                nuisance_rows = []
                for r in rows:
                    key = str(r.get("generator", r.get("domain", r.get("task_id", "unknown"))))
                    if key not in self._nuisance_to_id:
                        self._nuisance_to_id[key] = len(self._nuisance_to_id)
                    nuisance_rows.append(self._nuisance_to_id[key])
                nuisance = torch.tensor(nuisance_rows, dtype=torch.long) if nuisance_rows else None
                idx = hsic_guided_indices(feat, labels, nuisance, quota, lambda_kc=self.hgr_lambda_kc)
            self.memory.replace(list(self.memory.samples) + [rows[i] for i in idx])
        self.teacher = None if self.official_equivalent else self.frozen_detector_copy()
