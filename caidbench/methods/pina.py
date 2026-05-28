from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from ..registry import register_method
from .base import ContinualMethod, batch_to_device, build_optimizer, freeze_module


def _local_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = y.long()
    if y.numel() and (int(y.min()) < 0):
        raise ValueError("Labels must be non-negative for domain-incremental methods.")
    return torch.remainder(y, int(num_classes))


def _task_id(task: Any) -> int:
    return int(getattr(task, "task_id", task if isinstance(task, int) else 0))


def _official_optimizer_cfg(
    optimizer_cfg: dict[str, Any] | None,
    task_id: int,
    *,
    init_lr: float | None = None,
    lr: float | None = None,
    lrate: float | None = None,
    init_weight_decay: float | None = None,
    weight_decay: float | None = None,
    optimizer_type: str = "sgd",
) -> dict[str, Any]:
    cfg = dict(optimizer_cfg or {})
    cfg.setdefault("type", optimizer_type)
    cfg.setdefault("momentum", 0.9)
    official_lr = init_lr if int(task_id) == 0 and init_lr is not None else lr if lr is not None else lrate
    official_wd = init_weight_decay if int(task_id) == 0 and init_weight_decay is not None else weight_decay
    if official_lr is not None:
        cfg["lr"] = float(official_lr)
    if official_wd is not None:
        cfg["weight_decay"] = float(official_wd)
    return cfg


def _as_int_list(value: Sequence[int] | None) -> list[int]:
    return [int(v) for v in value] if value is not None else []


def _official_task_epochs(trainer: Any, task_id: int, init_epoch: int | None, epochs: int | None) -> int:
    if int(task_id) == 0 and init_epoch is not None:
        return max(int(init_epoch), 1)
    if epochs is not None:
        return max(int(epochs), 1)
    return max(int(getattr(trainer, "max_epochs", 1)), 1)


def _official_scheduler(
    optimizer: torch.optim.Optimizer,
    task_id: int,
    *,
    epochs: int | None = None,
    init_milestones: Sequence[int] | None = None,
    milestones: Sequence[int] | None = None,
    init_lr_decay: float | None = None,
    lr_decay: float | None = None,
    lrate_decay: float | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    points = _as_int_list(init_milestones if int(task_id) == 0 and init_milestones is not None else milestones)
    gamma = init_lr_decay if int(task_id) == 0 and init_lr_decay is not None else lrate_decay if lrate_decay is not None else lr_decay
    if not points or gamma is None:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs or 1), 1))
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=points, gamma=float(gamma))


def _run_minibatch_loop(
    method: ContinualMethod,
    trainer: Any,
    task: Any,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    method.train()
    for epoch in range(int(epochs)):
        totals: dict[str, float] = {}
        n = 0
        for batch in train_loader:
            out = method.observe(batch, task)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            method.transform_gradients(task)
            if trainer.grad_clip:
                torch.nn.utils.clip_grad_norm_(method.parameters(), trainer.grad_clip)
            optimizer.step()
            trainer.advance_step()
            method.after_optimizer_step(task)
            for key, value in out.items():
                if key == "logits":
                    continue
                if torch.is_tensor(value) and value.ndim == 0:
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            n += 1
        if scheduler is not None:
            scheduler.step()
        if totals:
            metrics = {key: value / max(n, 1) for key, value in totals.items()}
            msg = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            trainer.logger.info("task=%s epoch=%d/%d %s", task.name, epoch + 1, epochs, msg)
            trainer.log_metrics(
                {
                    **{f"train/{key}": value for key, value in metrics.items()},
                    "train/task_index": float(_task_id(task)),
                    "train/epoch": epoch + 1,
                }
            )


class FrozenFeatureMethod(ContinualMethod):
    """Common frozen-detector feature path for domain-incremental reproductions."""

    def __init__(self, freeze_backbone: bool = True, normalize_features: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.freeze_backbone = bool(freeze_backbone)
        self.normalize_features = bool(normalize_features)
        if self.freeze_backbone:
            freeze_module(self.detector.backbone)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_backbone:
            self.detector.backbone.eval()
        return self

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.detector.extract_features(x.to(self.device))
        z = z.float()
        return F.normalize(z, dim=-1) if self.normalize_features else z

    @torch.no_grad()
    def collect_features(self, loader: Any) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            features.append(self.extract_features(batch["x"]).detach().cpu())
            labels.append(_local_targets(batch["y"].detach().cpu(), self.num_classes))
        if was_training:
            self.train()
        if not features:
            return torch.empty(0, int(self.detector.feature_dim)), torch.empty(0, dtype=torch.long)
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64, depth: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = int(dim)
        for _ in range(max(1, int(depth))):
            layers.extend([nn.Linear(width, int(hidden_dim)), nn.ReLU(inplace=True), nn.Linear(int(hidden_dim), int(dim))])
            width = int(dim)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DomainRoutedFeatureMethod(FrozenFeatureMethod):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_centers: int = 5,
        adapter_depth: int = 1,
        train_backbone: bool = False,
        init_lr: float | None = None,
        lr: float | None = None,
        lrate: float | None = None,
        init_weight_decay: float | None = None,
        weight_decay: float | None = None,
        optimizer_type: str = "sgd",
        init_epoch: int | None = None,
        epochs: int | None = None,
        init_milestones: Sequence[int] | None = None,
        milestones: Sequence[int] | None = None,
        init_lr_decay: float | None = None,
        lr_decay: float | None = None,
        lrate_decay: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(freeze_backbone=not bool(train_backbone), **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_centers = int(num_centers)
        self.adapter_depth = int(adapter_depth)
        self.init_lr = None if init_lr is None else float(init_lr)
        self.lr = None if lr is None else float(lr)
        self.lrate = None if lrate is None else float(lrate)
        self.init_weight_decay = None if init_weight_decay is None else float(init_weight_decay)
        self.weight_decay = None if weight_decay is None else float(weight_decay)
        self.optimizer_type = str(optimizer_type)
        self.init_epoch = None if init_epoch is None else int(init_epoch)
        self.epochs = None if epochs is None else int(epochs)
        self.init_milestones = _as_int_list(init_milestones)
        self.milestones = _as_int_list(milestones)
        self.init_lr_decay = None if init_lr_decay is None else float(init_lr_decay)
        self.lr_decay = None if lr_decay is None else float(lr_decay)
        self.lrate_decay = None if lrate_decay is None else float(lrate_decay)
        self.adapters = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        self.current_key = "task0"

    def _key(self, task_id: int) -> str:
        return f"task{int(task_id)}"

    def _center_name(self, key: str) -> str:
        return f"{self.method_name}_centers_{key}"

    def _ensure_task_modules(self, key: str) -> None:
        if key in self.adapters:
            return
        dim = int(self.detector.feature_dim)
        self.adapters[key] = ResidualAdapter(dim, hidden_dim=self.hidden_dim, depth=self.adapter_depth)
        self.classifiers[key] = nn.Linear(dim, self.num_classes)

    def _freeze_except_current(self) -> None:
        for key, module in self.adapters.items():
            for p in module.parameters():
                p.requires_grad_(key == self.current_key)
        for key, module in self.classifiers.items():
            for p in module.parameters():
                p.requires_grad_(key == self.current_key)

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        super().before_task(task, train_loader)
        self.current_key = self._key(_task_id(task))
        self._ensure_task_modules(self.current_key)
        self._freeze_except_current()

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        task_id = 0 if self.current_task_id is None else int(self.current_task_id)
        cfg = _official_optimizer_cfg(
            optimizer_cfg,
            task_id,
            init_lr=self.init_lr,
            lr=self.lr,
            lrate=self.lrate,
            init_weight_decay=self.init_weight_decay,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
        )
        cfg.setdefault("lr", 1e-3)
        return build_optimizer(self.parameters(), cfg)

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        task_id = _task_id(task)
        epochs = _official_task_epochs(trainer, task_id, self.init_epoch, self.epochs)
        optimizer = self.configure_optimizer(trainer.optimizer_cfg)
        scheduler = _official_scheduler(
            optimizer,
            task_id,
            epochs=epochs,
            init_milestones=self.init_milestones,
            milestones=self.milestones,
            init_lr_decay=self.init_lr_decay,
            lr_decay=self.lr_decay,
            lrate_decay=self.lrate_decay,
        )
        _run_minibatch_loop(self, trainer, task, train_loader, optimizer, epochs, scheduler)
        return True

    @torch.no_grad()
    def _store_centers(self, key: str, loader: Any) -> None:
        features, _labels = self.collect_features(loader)
        features = F.normalize(features.float(), dim=-1)
        if features.numel() == 0:
            centers = features.reshape(0, int(self.detector.feature_dim))
        else:
            unique = np.unique(features.cpu().numpy(), axis=0)
            n_clusters = min(max(self.num_centers, 1), len(unique))
            if n_clusters == 1:
                centers_np = unique[:1]
            else:
                try:
                    centers_np = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(unique).cluster_centers_
                except TypeError:
                    centers_np = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(unique).cluster_centers_
            centers = torch.as_tensor(centers_np, dtype=torch.float32)
        name = self._center_name(key)
        if name in self._buffers:
            self._buffers[name] = centers.to(self.device)
        else:
            self.register_buffer(name, centers.to(self.device))

    def _route(self, z: torch.Tensor) -> torch.Tensor:
        keys = list(self.adapters.keys())
        if not keys:
            return torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        distances = []
        routed_keys = []
        z_norm = F.normalize(z.float(), dim=-1)
        for key in keys:
            name = self._center_name(key)
            if name not in self._buffers:
                continue
            centers = getattr(self, name).to(z_norm.device)
            if centers.numel() == 0:
                continue
            distances.append((z_norm[:, None, :] - centers[None, :, :]).abs().sum(dim=-1).min(dim=1).values)
            routed_keys.append(key)
        if not distances:
            return torch.full((z.shape[0],), len(keys) - 1, dtype=torch.long, device=z.device)
        chosen = torch.stack(distances, dim=1).argmin(dim=1)
        key_to_idx = {key: i for i, key in enumerate(keys)}
        return torch.tensor([key_to_idx[routed_keys[int(i)]] for i in chosen.detach().cpu()], dtype=torch.long, device=z.device)

    def _task_logits(self, z: torch.Tensor, key: str) -> torch.Tensor:
        return self.classifiers[key](self.adapters[key](z))

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.extract_features(x)
        keys = list(self.adapters.keys())
        selection = self._route(z)
        logits_by_task = torch.stack([self._task_logits(z, key) for key in keys], dim=1)
        logits = logits_by_task[torch.arange(z.shape[0], device=z.device), selection]
        return {"logits": logits, "features": z, "task_selection": selection.detach()}

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        z = self.extract_features(batch["x"])
        logits = self._task_logits(z, self.current_key)
        ce = F.cross_entropy(logits, _local_targets(batch["y"], self.num_classes))
        return {"loss": ce, "ce": ce.detach()}

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is not None:
            self._store_centers(self.current_key, train_loader)
        self._freeze_except_current()


@register_method("pina")
class PINAMethod(DomainRoutedFeatureMethod):
    """PINA/PINA-D feature-adapter reproduction with official UC/DSA/PSS behavior."""

    def __init__(self, ca_mode: str = "shallow", train_unified_on_first_task: bool = True, **kwargs: Any) -> None:
        depth = 1 if str(ca_mode).lower() == "shallow" else 2
        super().__init__(adapter_depth=depth, **kwargs)
        self.ca_mode = str(ca_mode).lower()
        self.train_unified_on_first_task = bool(train_unified_on_first_task)
        self.unified_classifier = nn.Linear(int(self.detector.feature_dim), self.num_classes)

    def _ensure_task_modules(self, key: str) -> None:
        if key in self.adapters:
            return
        dim = int(self.detector.feature_dim)
        self.adapters[key] = ResidualAdapter(dim, hidden_dim=self.hidden_dim, depth=self.adapter_depth)
        self.classifiers[key] = self.unified_classifier

    def _freeze_except_current(self) -> None:
        for key, module in self.adapters.items():
            for p in module.parameters():
                p.requires_grad_(key == self.current_key)
        for p in self.unified_classifier.parameters():
            p.requires_grad_(self.current_task_id == 0 and self.train_unified_on_first_task)

    def _task_logits(self, z: torch.Tensor, key: str) -> torch.Tensor:
        return self.unified_classifier(self.adapters[key](z))


@register_method("pina_d")
@register_method("pina-d")
class PINADMethod(PINAMethod):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("ca_mode", "deep")
        super().__init__(**kwargs)
