from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F

from ..memory import ReplayBuffer
from ..models.ekfn import ExpertKnowledgeFusionNetwork
from ..models.heads import Detector
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, freeze_module, merge_batches


@register_method("e3")
class E3Method(ContinualMethod):
    """Ensemble of Expert Embedders with EKFN fusion.

    The method trains a baseline detector, then trains one detector copy per new
    generator/domain and stores only its embedder. Final predictions pass an
    image through all frozen expert embedders and fuse their embeddings with a
    Transformer-based Expert Knowledge Fusion Network.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        memory_batch_size: int = 32,
        replay_group_key: str = "label",
        ekfn_layers: int = 2,
        ekfn_heads: int = 4,
        ekfn_hidden: int = 512,
        ekfn_dropout: float = 0.0,
        ekfn_activation: str = "gelu",
        max_experts: int = 64,
        expert_epochs: int | None = None,
        ekfn_epochs: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.memory = ReplayBuffer(memory_size, balanced=True, group_key=replay_group_key)
        self.memory_batch_size = int(memory_batch_size)
        self.expert_epochs = expert_epochs
        self.ekfn_epochs = ekfn_epochs
        self.experts = torch.nn.ModuleList()
        self.ekfn = ExpertKnowledgeFusionNetwork(
            embed_dim=self.detector.feature_dim,
            max_experts=max_experts,
            num_classes=self.num_classes,
            transformer_layers=ekfn_layers,
            nhead=ekfn_heads,
            mlp_hidden=ekfn_hidden,
            dropout=ekfn_dropout,
            activation=ekfn_activation,
        )

    def _expert_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.experts) == 0:
            return self.detector.extract_features(x).unsqueeze(1)
        feats = []
        for expert in self.experts:
            feats.append(expert(x).detach())
        return torch.stack(feats, dim=1)

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        if len(self.experts) <= 1:
            return self.detector(x)
        emb = self._expert_embeddings(x)
        logits = self.ekfn(emb)
        fused = emb.mean(dim=1)
        return {"logits": logits, "features": fused}

    def _train_detector_copy(self, model: Detector, loader: Any, trainer: Any, task: Any, epochs: int) -> None:
        model.to(self.device).train()
        opt = trainer.make_optimizer(model.parameters())
        for _ in range(epochs):
            for batch in loader:
                batch = batch_to_device(batch, self.device)
                mem = self.memory.sample(self.memory_batch_size, device=self.device)
                train_batch = merge_batches(batch, mem)
                out = model(train_batch["x"])
                loss = F.cross_entropy(out["logits"], train_batch["y"].long())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.grad_clip)
                opt.step()
                trainer.advance_step()

    def _train_ekfn(self, loader: Any, trainer: Any, epochs: int) -> None:
        if len(self.experts) == 0:
            return
        self.ekfn.to(self.device).train()
        opt = trainer.make_optimizer(self.ekfn.parameters())
        for expert in self.experts:
            freeze_module(expert)
        for _ in range(epochs):
            for batch in loader:
                batch = batch_to_device(batch, self.device)
                mem = self.memory.sample(self.memory_batch_size, device=self.device)
                train_batch = merge_batches(batch, mem)
                emb = self._expert_embeddings(train_batch["x"])
                logits = self.ekfn(emb)
                loss = F.cross_entropy(logits, train_batch["y"].long())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ekfn.parameters(), trainer.grad_clip)
                opt.step()
                trainer.advance_step()

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        epochs = int(trainer.max_epochs)
        expert_epochs = int(self.expert_epochs or epochs)
        ekfn_epochs = int(self.ekfn_epochs or epochs)
        if len(self.experts) == 0:
            # Initial baseline detector training.
            trainer.default_train_loop(self, task, train_loader)
            self.experts.append(freeze_module(copy.deepcopy(self.detector.backbone).to(self.device)))
        else:
            # Train an expert detector for the incoming generator/task and keep only its embedder.
            expert_detector = copy.deepcopy(self.detector)
            self._train_detector_copy(expert_detector, train_loader, trainer, task, expert_epochs)
            self.experts.append(freeze_module(copy.deepcopy(expert_detector.backbone).to(self.device)))
            self._train_ekfn(train_loader, trainer, ekfn_epochs)
        for batch in train_loader:
            self.memory.add_batch(batch)
        return True
