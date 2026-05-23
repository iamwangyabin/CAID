from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..models.adapters import MultiSceneLoRAHead
from ..registry import register_method
from .base import ContinualMethod, batch_to_device


@register_method("saido")
class SAIDOMethod(ContinualMethod):
    """Scene-Aware and Importance-Guided Dynamic Optimization.

    SAEM is represented by scene-routed LoRA classifier heads over a shared
    visual backbone. IDOM is represented by Fisher-style neuron/parameter
    importance estimates and gradient scaling/projection hooks.
    """

    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        freeze_backbone: bool = False,
        importance_quantile: float = 0.8,
        core_grad_scale: float = 0.1,
        noncore_grad_scale: float = 1.0,
        contrast_weight: float = 0.05,
        scene_prompt_weight: float = 0.0,
        scene_temperature: float = 0.07,
        idom_projection: bool = False,
        old_grad_scale: float = 1.0,
        importance_batches: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scene_head = MultiSceneLoRAHead(self.detector.feature_dim, num_classes=self.num_classes, rank=lora_rank, alpha=lora_alpha)
        self.detector.head.requires_grad_(False)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.detector.backbone.parameters():
                p.requires_grad_(False)
        self.importance_quantile = float(importance_quantile)
        self.core_grad_scale = float(core_grad_scale)
        self.noncore_grad_scale = float(noncore_grad_scale)
        self.contrast_weight = float(contrast_weight)
        self.scene_prompt_weight = float(scene_prompt_weight)
        self.scene_temperature = float(scene_temperature)
        self.idom_projection = bool(idom_projection)
        self.old_grad_scale = float(old_grad_scale)
        self.importance_batches = int(importance_batches)
        self.importance: dict[str, torch.Tensor] = {}
        self.gradient_memory: dict[str, torch.Tensor] = {}

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        super().before_task(task, train_loader)
        for s in getattr(task, "scenes", ()) or ("unknown",):
            self.scene_head.add_scene(str(s))
        self.scene_head.to(self.device)

    def _scenes(self, batch: dict[str, Any]) -> list[str]:
        scenes = batch.get("scene")
        if scenes is None:
            return ["unknown"] * int(batch["x"].shape[0])
        if torch.is_tensor(scenes):
            return [str(int(x)) for x in scenes.detach().cpu().view(-1).tolist()]
        return [str(s) for s in scenes]

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.detector.extract_features(x)
        logits = self.scene_head(z, self._scenes(batch))
        return {"logits": logits, "features": z}

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        out = self.predict(batch)
        y = batch["y"].long()
        ce = F.cross_entropy(out["logits"], y)
        # Scene-aware compactness term: stabilize same-scene real/fake evidence.
        z = F.normalize(out["features"], dim=-1)
        scene_ids: dict[str, int] = {}
        sid = []
        for s in self._scenes(batch):
            scene_ids.setdefault(s, len(scene_ids))
            sid.append(scene_ids[s])
        sid_t = torch.tensor(sid, device=self.device)
        contrast = z.new_tensor(0.0)
        for s in torch.unique(sid_t):
            m = sid_t == s
            if m.sum() > 1:
                contrast = contrast + (z[m] - z[m].mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean()
        prompt_contrast = self._scene_prompt_contrast(z, batch)
        loss = ce + self.contrast_weight * contrast + self.scene_prompt_weight * prompt_contrast
        return {"loss": loss, "ce": ce.detach(), "scene_compact": contrast.detach(), "scene_prompt": prompt_contrast.detach()}

    def _scene_prompt_contrast(self, image_features: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        prompt_features = batch.get("scene_prompt_features", batch.get("scene_text_features"))
        if prompt_features is None or not torch.is_tensor(prompt_features):
            return image_features.new_tensor(0.0)
        prompt_features = prompt_features.to(device=image_features.device, dtype=image_features.dtype)
        if prompt_features.ndim != 2 or prompt_features.shape != image_features.shape:
            raise ValueError(
                "SAIDO scene_prompt_features must be a tensor with the same [B,D] shape as image features; "
                f"got {tuple(prompt_features.shape)} vs {tuple(image_features.shape)}."
            )
        img = F.normalize(image_features, dim=-1)
        txt = F.normalize(prompt_features, dim=-1)
        logits = img @ txt.T / max(self.scene_temperature, 1e-6)
        target = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

    def transform_gradients(self, task: Any | None = None) -> None:
        if not self.importance:
            return
        for name, p in self.named_parameters():
            if p.grad is None or name not in self.importance:
                continue
            if self.idom_projection and name in self.gradient_memory:
                ref = self.gradient_memory[name].to(device=p.grad.device, dtype=p.grad.dtype)
                denom = ref.pow(2).sum().clamp_min(1e-12)
                parallel = (p.grad * ref).sum() / denom * ref
                orthogonal = p.grad - parallel
                p.grad.copy_(orthogonal + self.old_grad_scale * parallel)
            imp = self.importance[name].to(p.grad.device)
            if imp.numel() == 0:
                continue
            threshold = torch.quantile(imp.flatten(), self.importance_quantile)
            core = (imp >= threshold).to(p.grad.dtype)
            scale = core * self.core_grad_scale + (1.0 - core) * self.noncore_grad_scale
            p.grad.mul_(scale)

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is None:
            return
        accum: dict[str, torch.Tensor] = {}
        grad_accum: dict[str, torch.Tensor] = {}
        self.train()
        for i, batch in enumerate(train_loader):
            if i >= self.importance_batches:
                break
            batch = batch_to_device(batch, self.device)
            out = self.predict(batch)
            loss = F.cross_entropy(out["logits"], batch["y"].long())
            self.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in self.named_parameters():
                if p.grad is None:
                    continue
                accum[name] = accum.get(name, torch.zeros_like(p.grad.detach())) + p.grad.detach().pow(2).cpu()
                grad_accum[name] = grad_accum.get(name, torch.zeros_like(p.grad.detach())) + p.grad.detach().cpu()
        self.zero_grad(set_to_none=True)
        if accum:
            for k, v in accum.items():
                old = self.importance.get(k)
                self.importance[k] = v if old is None else 0.5 * old + 0.5 * v
        if grad_accum:
            for k, v in grad_accum.items():
                norm = v.norm().clamp_min(1e-12)
                old = self.gradient_memory.get(k)
                direction = v / norm
                mixed = direction if old is None else 0.5 * old + 0.5 * direction
                self.gradient_memory[k] = mixed / mixed.norm().clamp_min(1e-12)
