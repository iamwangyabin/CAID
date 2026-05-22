from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..losses import category_alignment_loss, feature_distillation_loss, kd_loss, pairwise_distance_mse
from ..models.adapters import AdapterBlock, grid_shuffle
from ..registry import register_method
from .base import ContinualMethod, batch_to_device


class AdapterDetector(nn.Module):
    def __init__(self, detector: nn.Module, bottleneck: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.detector = detector
        self.adapter = AdapterBlock(detector.feature_dim, bottleneck=bottleneck, scale=scale)
        self.feature_dim = detector.feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.detector.extract_features(x)
        return self.adapter(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.extract_features(x)
        logits = self.detector.head(z)
        return {"logits": logits, "features": z}


@register_method("ca_adapter_cail")
@register_method("content_agnostic_adapter")
class ContentAgnosticAdapterCAIL(ContinualMethod):
    """Content-agnostic adapter with category-aware incremental alignment.

    Exposes the mechanisms described in the TIFS work: ViT/CNN feature adapter,
    content perturbation through token/grid shuffling, asymmetric category-aware
    alignment, and point/structure-level knowledge distillation.
    """

    def __init__(
        self,
        adapter_bottleneck: int = 64,
        adapter_scale: float = 1.0,
        shuffle_grid: int = 4,
        shuffle_weight: float = 0.5,
        align_weight: float = 0.1,
        kd_weight: float = 1.0,
        structure_kd_weight: float = 0.5,
        temperature: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.adapter_detector = AdapterDetector(self.detector, bottleneck=adapter_bottleneck, scale=adapter_scale)
        self.detector = self.adapter_detector
        self.shuffle_grid = int(shuffle_grid)
        self.shuffle_weight = float(shuffle_weight)
        self.align_weight = float(align_weight)
        self.kd_weight = float(kd_weight)
        self.structure_kd_weight = float(structure_kd_weight)
        self.temperature = float(temperature)
        self.teacher: nn.Module | None = None
        self._domain_to_id: dict[str, int] = {}

    def _domain_ids(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor | None:
        domains = batch.get("generator") or batch.get("domain")
        if domains is None or torch.is_tensor(domains):
            return domains.to(device) if torch.is_tensor(domains) else None
        ids = []
        for d in domains:
            key = str(d)
            if key not in self._domain_to_id:
                self._domain_to_id[key] = len(self._domain_to_id)
            ids.append(self._domain_to_id[key])
        return torch.tensor(ids, dtype=torch.long, device=device)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        y = batch["y"].long()
        out = self.predict(batch)
        ce = F.cross_entropy(out["logits"], y)
        shuffled_x = grid_shuffle(batch["x"], grid=self.shuffle_grid)
        shuf_out = self.detector(shuffled_x)
        shuf_ce = F.cross_entropy(shuf_out["logits"], y)
        content_consistency = F.mse_loss(F.normalize(out["features"], dim=-1), F.normalize(shuf_out["features"], dim=-1))
        domains = self._domain_ids(batch, self.device)
        align = category_alignment_loss(out["features"], y, domains=domains)
        loss = ce + self.shuffle_weight * (shuf_ce + content_consistency) + self.align_weight * align
        log: dict[str, torch.Tensor] = {"ce": ce.detach(), "shuffle_ce": shuf_ce.detach(), "align": align.detach()}
        if self.teacher is not None:
            with torch.no_grad():
                t_out = self.teacher(batch["x"])
            kd = kd_loss(out["logits"], t_out["logits"], self.temperature)
            p2p = feature_distillation_loss(out["features"], t_out["features"])
            s2s = pairwise_distance_mse(out["features"], t_out["features"])
            loss = loss + self.kd_weight * (kd + p2p) + self.structure_kd_weight * s2s
            log.update({"kd": kd.detach(), "point_kd": p2p.detach(), "structure_kd": s2s.detach()})
        log["loss"] = loss
        return log

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.teacher = self.frozen_detector_copy()
