from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from ..data.loader import build_dataloader
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, build_optimizer, freeze_module


def _local_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = y.long()
    if y.numel() and (int(y.min()) < 0):
        raise ValueError("Labels must be non-negative for official DIL methods.")
    return torch.remainder(y, int(num_classes))


def _task_id(task: Any) -> int:
    return int(getattr(task, "task_id", task if isinstance(task, int) else 0))


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items() if torch.is_floating_point(v)}


def _state_delta(current: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in current.items():
        if key in base and value.shape == base[key].shape:
            out[key] = value.detach().cpu() - base[key]
    return out


def _load_float_state(module: nn.Module, float_state: dict[str, torch.Tensor]) -> None:
    state = module.state_dict()
    for key, value in float_state.items():
        if key in state and state[key].shape == value.shape:
            state[key] = value.to(device=state[key].device, dtype=state[key].dtype)
    module.load_state_dict(state, strict=False)


def _one_hot(y: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return F.one_hot(y.long(), num_classes=int(num_classes)).to(dtype=dtype)


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
    init_milestones: Sequence[int] | None = None,
    milestones: Sequence[int] | None = None,
    init_lr_decay: float | None = None,
    lr_decay: float | None = None,
    lrate_decay: float | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    points = _as_int_list(init_milestones if int(task_id) == 0 and init_milestones is not None else milestones)
    gamma = init_lr_decay if int(task_id) == 0 and init_lr_decay is not None else lrate_decay if lrate_decay is not None else lr_decay
    if not points or gamma is None:
        return None
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
    """Common frozen-detector feature path for official DIL reproductions."""

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


class RidgeAccumulator(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.dtype = dtype
        self.register_buffer("gram", torch.zeros(self.feature_dim, self.feature_dim, dtype=dtype))
        self.register_buffer("cross", torch.zeros(self.feature_dim, self.num_classes, dtype=dtype))
        self.register_buffer("weight", torch.zeros(self.num_classes, self.feature_dim, dtype=torch.float32))
        self.ridge = 1.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x64 = x.detach().to(device=self.gram.device, dtype=self.dtype)
        y64 = _one_hot(y.to(self.gram.device), self.num_classes, dtype=self.dtype)
        self.gram += x64.t().matmul(x64)
        self.cross += x64.t().matmul(y64)

    def solve(self, ridge: float | None = None) -> torch.Tensor:
        value = float(self.ridge if ridge is None else ridge)
        eye = torch.eye(self.feature_dim, device=self.gram.device, dtype=self.dtype)
        solution = torch.linalg.solve(self.gram + value * eye, self.cross)
        self.weight = solution.t().float()
        self.ridge = value
        return self.weight

    @torch.no_grad()
    def select_ridge(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        candidates: Sequence[float],
        val_fraction: float = 0.2,
        include_history: bool = True,
    ) -> float:
        if x.shape[0] < 4 or len(candidates) <= 1:
            return float(candidates[0] if candidates else self.ridge)
        n_val = max(int(round(x.shape[0] * float(val_fraction))), 1)
        n_train = max(x.shape[0] - n_val, 1)
        x_train = x[:n_train].to(dtype=self.dtype, device=self.gram.device)
        y_train = _one_hot(y[:n_train].to(self.gram.device), self.num_classes, dtype=self.dtype)
        x_val = x[n_train:].to(dtype=self.dtype, device=self.gram.device)
        y_val = _one_hot(y[n_train:].to(self.gram.device), self.num_classes, dtype=self.dtype)
        if x_val.shape[0] == 0:
            return float(candidates[0])

        if include_history:
            g = self.gram + x_train.t().matmul(x_train)
            c = self.cross + x_train.t().matmul(y_train)
        else:
            g = x_train.t().matmul(x_train)
            c = x_train.t().matmul(y_train)
        eye = torch.eye(self.feature_dim, device=self.gram.device, dtype=self.dtype)
        best = float(candidates[0])
        best_loss = float("inf")
        for candidate in candidates:
            try:
                w = torch.linalg.solve(g + float(candidate) * eye, c)
            except RuntimeError:
                continue
            loss = F.mse_loss(x_val.matmul(w), y_val).item()
            if loss < best_loss:
                best_loss = loss
                best = float(candidate)
        return best

    @torch.no_grad()
    def select_ridge_stratified_accuracy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        candidates: Sequence[float],
        n_splits: int = 4,
    ) -> float:
        """Select ridge like official LayUP: stratified CV, maximize accuracy."""
        if x.shape[0] < 4 or len(candidates) <= 1:
            return float(candidates[0] if candidates else self.ridge)
        y_cpu = y.detach().long().cpu()
        labels_np = y_cpu.numpy()
        _classes, counts = np.unique(labels_np, return_counts=True)
        folds = min(int(n_splits), int(counts.min()) if len(counts) else 0)
        if folds < 2:
            return float(candidates[0] if candidates else self.ridge)

        try:
            from sklearn.model_selection import StratifiedKFold
        except Exception:
            return float(candidates[0] if candidates else self.ridge)

        x_all = x.detach().to(dtype=self.dtype, device=self.gram.device)
        y_onehot = _one_hot(y.to(self.gram.device), self.num_classes, dtype=self.dtype)
        eye = torch.eye(self.feature_dim, device=self.gram.device, dtype=self.dtype)
        accuracies = np.zeros(len(candidates), dtype=float)
        split_count = 0
        splitter = StratifiedKFold(n_splits=folds, shuffle=False)
        for train_idx_np, val_idx_np in splitter.split(np.zeros(len(labels_np)), labels_np):
            train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=self.gram.device)
            val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=self.gram.device)
            x_train = x_all.index_select(0, train_idx)
            y_train = y_onehot.index_select(0, train_idx)
            x_val = x_all.index_select(0, val_idx)
            y_val = y.to(self.gram.device).long().index_select(0, val_idx)
            g = self.gram + x_train.t().matmul(x_train)
            c = self.cross + x_train.t().matmul(y_train)
            for idx, candidate in enumerate(candidates):
                try:
                    w = torch.linalg.solve(g + float(candidate) * eye, c)
                except RuntimeError:
                    continue
                pred = x_val.matmul(w).argmax(dim=1)
                accuracies[idx] += float((pred == y_val).float().mean().item())
            split_count += 1

        if split_count == 0:
            return float(candidates[0] if candidates else self.ridge)
        accuracies /= float(split_count)
        return float(candidates[int(np.argmax(accuracies))])


class RandomProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_relu: bool = True) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.use_relu = bool(use_relu)
        if self.out_dim > 0:
            self.register_buffer("matrix", torch.randn(self.in_dim, self.out_dim))
        else:
            self.register_buffer("matrix", torch.empty(self.in_dim, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.out_dim <= 0:
            return x
        h = x.matmul(self.matrix.to(device=x.device, dtype=x.dtype))
        return F.relu(h) if self.use_relu else h


@register_method("ranpac")
class RanPACMethod(FrozenFeatureMethod):
    """RanPAC random projection plus ridge classifier following the official code path."""

    def __init__(
        self,
        M: int = 10000,
        use_RP: bool = True,
        ridge_candidates: Sequence[float] | None = None,
        ridge: float | None = None,
        ridge_val_fraction: float = 0.2,
        use_relu: bool = True,
        model_name: str = "ncm",
        tuned_epoch: int = 0,
        body_lr: float = 0.01,
        head_lr: float | None = None,
        weight_decay: float = 0.0005,
        min_lr: float = 0.0,
        adapter_bottleneck: int = 64,
        adapter_dropout: float = 0.0,
        adapter_scalar: float = 0.1,
        use_test_transform_for_ridge: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(freeze_backbone=True, **kwargs)
        self.M = int(M)
        self.use_RP = bool(use_RP)
        self.ridge_candidates = list(ridge_candidates or [10.0**i for i in range(-8, 9)])
        self.fixed_ridge = None if ridge is None else float(ridge)
        self.ridge_val_fraction = float(ridge_val_fraction)
        self.model_name = str(model_name).lower()
        self.tuned_epoch = int(tuned_epoch)
        self.body_lr = float(body_lr)
        self.head_lr = None if head_lr is None else float(head_lr)
        self.weight_decay = float(weight_decay)
        self.min_lr = float(min_lr)
        self.use_test_transform_for_ridge = bool(use_test_transform_for_ridge)
        self.adapter_tuned = False
        if self.tuned_epoch > 0 and self.model_name != "adapter":
            raise ValueError("RanPAC first-task tuning currently supports model_name='adapter', matching the official CDDB full setting.")
        if self.model_name == "adapter":
            _install_layup_adapters(
                self.detector.backbone,
                bottleneck=int(adapter_bottleneck),
                dropout=float(adapter_dropout),
                scalar=float(adapter_scalar),
            )
        feature_dim = int(self.detector.feature_dim)
        proj_dim = self.M if self.use_RP and self.M > 0 else feature_dim
        self.projector = RandomProjection(feature_dim, self.M if self.use_RP else 0, use_relu=use_relu)
        self.ridge_head = RidgeAccumulator(proj_dim, self.num_classes)

    def _build_ridge_loader(self, trainer: Any, task: Any, fallback_loader: Any) -> Any:
        if not self.use_test_transform_for_ridge:
            return fallback_loader
        scenario = getattr(trainer, "scenario", None)
        if scenario is None or not hasattr(scenario, "source"):
            return fallback_loader
        task_index = None
        for idx, spec in enumerate(getattr(scenario, "tasks", [])):
            if spec is task or (getattr(spec, "task_id", None) == getattr(task, "task_id", None) and getattr(spec, "name", None) == getattr(task, "name", None)):
                task_index = idx
                break
        if task_index is None:
            return fallback_loader
        split_indices = getattr(scenario, "_split_indices", {})
        indices = split_indices.get((task_index, "train"))
        if indices is None:
            return fallback_loader
        transform = scenario._transform_for_split("test")
        dataset = scenario.source.make_dataset(
            indices,
            transform_cfg=transform,
            task_id=getattr(task, "task_id", task_index),
            task_name=getattr(task, "name", f"task{task_index}"),
        )
        return build_dataloader(
            dataset,
            batch_size=int(getattr(trainer, "batch_size", 32)),
            shuffle=True,
            num_workers=int(getattr(trainer, "num_workers", 0)),
            drop_last=False,
        )

    def _train_first_task_adapter(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None) -> None:
        if self.tuned_epoch <= 0 or self.adapter_tuned or _task_id(task) != 0:
            return
        self.detector.train()
        self.detector.backbone.train()
        trainable = [p for p in self.detector.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("RanPAC adapter tuning has no trainable adapter/head parameters.")
        head_lr = self.body_lr if self.head_lr is None else self.head_lr
        backbone_params = [p for p in self.detector.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.detector.head.parameters() if p.requires_grad]
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.body_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr})
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.tuned_epoch, 1),
            eta_min=self.min_lr,
        )
        eval_loader = val_loader if val_loader is not None else train_loader
        for epoch in range(max(self.tuned_epoch, 1)):
            self.detector.train()
            self.detector.backbone.train()
            total_loss = 0.0
            correct = 0
            total = 0
            batches = 0
            for batch in train_loader:
                batch = batch_to_device(batch, self.device)
                logits = self.detector(batch["x"])["logits"]
                y = _local_targets(batch["y"], self.num_classes).to(logits.device)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                total_loss += float(loss.detach().cpu())
                correct += int((logits.detach().argmax(dim=1) == y).sum().item())
                total += int(y.numel())
                batches += 1
            scheduler.step()
            tune_acc = self._adapter_accuracy(eval_loader)
            train_acc = float(correct / max(total, 1))
            avg_loss = total_loss / max(batches, 1)
            trainer.logger.info(
                "task=%s ranpac_tune epoch=%d/%d loss=%.4f train_acc=%.4f eval_acc=%.4f",
                task.name,
                epoch + 1,
                self.tuned_epoch,
                avg_loss,
                train_acc,
                tune_acc,
            )
            trainer.log_metrics(
                {
                    "train/ranpac_tune_loss": avg_loss,
                    "train/ranpac_tune_acc": train_acc,
                    "train/ranpac_tune_eval_acc": tune_acc,
                    "train/task_index": float(_task_id(task)),
                    "train/epoch": epoch + 1,
                }
            )
        freeze_module(self.detector.backbone)
        freeze_module(self.detector.head)
        self.adapter_tuned = True

    @torch.no_grad()
    def _adapter_accuracy(self, loader: Any) -> float:
        was_training = self.detector.training
        self.detector.eval()
        correct = 0
        total = 0
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            logits = self.detector(batch["x"])["logits"]
            y = _local_targets(batch["y"], self.num_classes).to(logits.device)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
        if was_training:
            self.detector.train()
        return float(correct / max(total, 1))

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        self._train_first_task_adapter(trainer, task, train_loader, val_loader)
        ridge_loader = self._build_ridge_loader(trainer, task, train_loader)
        features, labels = self.collect_features(ridge_loader)
        features = features.to(self.device)
        labels = labels.to(self.device)
        h = self.projector(features)
        ridge = self.fixed_ridge
        if ridge is None:
            ridge = self.ridge_head.select_ridge(
                h,
                labels,
                self.ridge_candidates,
                self.ridge_val_fraction,
                include_history=False,
            )
        self.ridge_head.update(h, labels)
        self.ridge_head.solve(ridge)
        trainer.logger.info("task=%s ranpac_ridge=%.3g samples=%d dim=%d", task.name, ridge, labels.numel(), h.shape[1])
        trainer.log_metrics({"train/ranpac_ridge": ridge, "train/task_index": float(_task_id(task))})
        return True

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.extract_features(x)
        h = self.projector(z)
        logits = h.matmul(self.ridge_head.weight.to(device=h.device, dtype=h.dtype).t())
        return {"logits": logits, "features": z}


class MultiLayerFeatureExtractor(nn.Module):
    """Feature extractor that concatenates configured intermediate activations."""

    def __init__(self, detector: nn.Module, layer_names: Sequence[str] | None = None, token_pool: str = "cls") -> None:
        super().__init__()
        self.detector = detector
        self.layer_names = [str(name) for name in (layer_names or [])]
        self.token_pool = str(token_pool)

    @staticmethod
    def _pool(out: Any, token_pool: str) -> torch.Tensor:
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"Hooked layer returned {type(out)!r}, expected tensor")
        if out.ndim == 3:
            return out.mean(dim=1).float() if token_pool == "mean" else out[:, 0].float()
        if out.ndim == 4:
            return F.adaptive_avg_pool2d(out.float(), 1).flatten(1)
        return out.reshape(out.shape[0], -1).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.layer_names:
            return self.detector.extract_features(x).float()
        activations: dict[str, torch.Tensor] = {}
        handles = []
        for name in self.layer_names:
            module = self.detector.get_submodule(name)
            handles.append(module.register_forward_hook(lambda _m, _i, out, key=name: activations.setdefault(key, out)))
        try:
            final = self.detector.extract_features(x)
        finally:
            for handle in handles:
                handle.remove()
        parts = [self._pool(activations[name], self.token_pool) for name in sorted(activations)]
        if not parts:
            return final.float()
        return torch.cat(parts, dim=1).float()


class _LayUPAdapter(nn.Module):
    """AdaptFormer-style residual adapter used for LayUP first-session adaptation."""

    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.1, scalar: float = 0.1) -> None:
        super().__init__()
        self.down_proj = nn.Linear(int(dim), int(bottleneck))
        self.up_proj = nn.Linear(int(bottleneck), int(dim))
        self.dropout = float(dropout)
        self.scalar = float(scalar)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.down_proj(x))
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.scalar * self.up_proj(z)


class _LayUPAdapterBlock(nn.Module):
    """Thin wrapper that keeps the original ViT block and adds a trainable adapter branch."""

    def __init__(self, block: nn.Module, dim: int, bottleneck: int = 64, dropout: float = 0.1, scalar: float = 0.1) -> None:
        super().__init__()
        self.block = block
        self.adaptmlp = _LayUPAdapter(dim, bottleneck=bottleneck, dropout=dropout, scalar=scalar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        required = ("norm1", "attn", "drop_path1", "ls1", "norm2", "mlp", "drop_path2", "ls2")
        if not all(hasattr(self.block, name) for name in required):
            out = self.block(x)
            return out + self.adaptmlp(out)
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        adapt_x = self.adaptmlp(x)
        residual = x
        x = self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))
        return adapt_x + residual + x

    def freeze(self, fully: bool = False) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        if not fully:
            for param in self.adaptmlp.parameters():
                param.requires_grad_(True)


def _timm_inner_model(backbone: nn.Module) -> nn.Module | None:
    return getattr(backbone, "model", None)


def _install_layup_adapters(backbone: nn.Module, bottleneck: int = 64, dropout: float = 0.1, scalar: float = 0.1) -> None:
    model = _timm_inner_model(backbone)
    blocks = getattr(model, "blocks", None) if model is not None else None
    if blocks is None:
        raise RuntimeError("LayUP finetune_method=adapter requires a timm ViT backbone with a blocks module.")
    for idx, block in enumerate(blocks):
        if isinstance(block, _LayUPAdapterBlock):
            continue
        dim = getattr(block, "dim", None)
        if dim is None and hasattr(block, "attn"):
            attn = getattr(block, "attn")
            dim = int(getattr(attn, "num_heads")) * int(getattr(attn, "head_dim"))
        if dim is None:
            dim = int(getattr(model, "num_features", getattr(backbone, "out_dim")))
        blocks[idx] = _LayUPAdapterBlock(block, int(dim), bottleneck=bottleneck, dropout=dropout, scalar=scalar)


def _freeze_layup_backbone(backbone: nn.Module, fully: bool) -> None:
    model = _timm_inner_model(backbone)
    target = model if model is not None else backbone
    called = False
    for module in target.modules():
        freeze_fn = getattr(module, "freeze", None)
        if callable(freeze_fn):
            freeze_fn(fully=fully)
            called = True
    if not called:
        for param in backbone.parameters():
            param.requires_grad_(not fully)
    if fully:
        backbone.eval()


@register_method("layup")
class LayUPMethod(FrozenFeatureMethod):
    """LayUP multi-layer feature ridge classifier."""

    def __init__(
        self,
        k: int = 6,
        layer_names: Sequence[str] | None = None,
        ridge_candidates: Sequence[float] | None = None,
        ridge_splits: int = 4,
        token_pool: str = "cls",
        finetune_method: str = "none",
        finetune_epochs: int = 20,
        early_stopping: int = 5,
        lr: float = 0.003,
        weight_decay: float = 0.0005,
        adapter_bottleneck: int = 64,
        adapter_dropout: float = 0.1,
        adapter_scalar: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(freeze_backbone=True, **kwargs)
        self.finetune_method = str(finetune_method).lower()
        self.finetune_epochs = int(finetune_epochs)
        self.early_stopping = int(early_stopping)
        self.fsa_lr = float(lr)
        self.fsa_weight_decay = float(weight_decay)
        self.fsa_done = False
        if self.finetune_method in {"adaptformer", "adapter"}:
            _install_layup_adapters(
                self.detector.backbone,
                bottleneck=int(adapter_bottleneck),
                dropout=float(adapter_dropout),
                scalar=float(adapter_scalar),
            )
            freeze_module(self.detector.head)
            _freeze_layup_backbone(self.detector.backbone, fully=True)
        elif self.finetune_method not in {"none", "no", "false", "off"}:
            raise ValueError("LayUP currently supports finetune_method='none' or 'adapter'/'adaptformer'.")
        if layer_names is None:
            layer_names = self._default_vit_layers(int(k))
        self.layer_names = list(layer_names)
        self.feature_extractor = MultiLayerFeatureExtractor(self.detector, self.layer_names, token_pool=token_pool)
        self.ridge_candidates = list(ridge_candidates or [1e-8, *np.logspace(-4, 3, 15).tolist()])
        self.ridge_splits = int(ridge_splits)
        self.ridge_head: RidgeAccumulator | None = None

    def _default_vit_layers(self, k: int) -> list[str]:
        backbone = getattr(self.detector, "backbone", None)
        model = getattr(backbone, "model", None)
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            return []
        n_blocks = len(blocks)
        return [f"backbone.model.blocks.{n_blocks - 1 - i}" for i in range(min(max(k, 0), n_blocks))]

    @torch.no_grad()
    def collect_features(self, loader: Any) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        was_training = self.training
        self.eval()
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            features.append(self.feature_extractor(batch["x"]).detach().cpu())
            labels.append(_local_targets(batch["y"].detach().cpu(), self.num_classes))
        if was_training:
            self.train()
        if not features:
            dim = int(self.detector.feature_dim)
            return torch.empty(0, dim), torch.empty(0, dtype=torch.long)
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def _uses_fsa(self) -> bool:
        return self.finetune_method not in {"none", "no", "false", "off"}

    @torch.no_grad()
    def _evaluate_fsa_head(self, head: nn.Module, loader: Any) -> float:
        self.detector.backbone.eval()
        head.eval()
        correct = 0
        total = 0
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            z = self.detector.extract_features(batch["x"])
            logits = 30.0 * head(z)
            y = _local_targets(batch["y"], self.num_classes).to(logits.device)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
        return float(correct / max(total, 1))

    def _run_fsa(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None) -> None:
        if not self._uses_fsa() or self.fsa_done or _task_id(task) != 0:
            return
        _freeze_layup_backbone(self.detector.backbone, fully=False)
        head = CosineLinear(int(self.detector.feature_dim), self.num_classes).to(self.device)
        trainable = [p for p in self.detector.backbone.parameters() if p.requires_grad] + list(head.parameters())
        if not trainable:
            raise RuntimeError("LayUP FSA has no trainable PETL/head parameters.")
        optimizer = torch.optim.AdamW(trainable, lr=self.fsa_lr, weight_decay=self.fsa_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.finetune_epochs, 1), eta_min=0.0)
        eval_loader = val_loader if val_loader is not None else train_loader
        best_acc = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        for epoch in range(max(self.finetune_epochs, 1)):
            self.detector.backbone.train()
            head.train()
            total_loss = 0.0
            total_correct = 0
            total = 0
            batches = 0
            for batch in train_loader:
                batch = batch_to_device(batch, self.device)
                z = self.detector.extract_features(batch["x"])
                logits = 30.0 * head(z)
                y = _local_targets(batch["y"], self.num_classes).to(logits.device)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                trainer.advance_step()
                total_loss += float(loss.detach().cpu())
                total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
                total += int(y.numel())
                batches += 1
            scheduler.step()
            val_acc = self._evaluate_fsa_head(head, eval_loader)
            train_acc = float(total_correct / max(total, 1))
            trainer.logger.info(
                "task=%s fsa_epoch=%d/%d loss=%.4f train_acc=%.4f val_acc=%.4f",
                task.name,
                epoch + 1,
                max(self.finetune_epochs, 1),
                total_loss / max(batches, 1),
                train_acc,
                val_acc,
            )
            trainer.log_metrics(
                {
                    "train/layup_fsa_loss": total_loss / max(batches, 1),
                    "train/layup_fsa_acc": train_acc,
                    "train/layup_fsa_val_acc": val_acc,
                    "train/task_index": float(_task_id(task)),
                    "train/epoch": epoch + 1,
                }
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.detector.backbone.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= max(self.early_stopping, 1):
                break
        if best_state is not None:
            self.detector.backbone.load_state_dict(best_state, strict=False)
        _freeze_layup_backbone(self.detector.backbone, fully=True)
        self.detector.backbone.eval()
        self.fsa_done = True

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        self._run_fsa(trainer, task, train_loader, val_loader)
        x, y = self.collect_features(train_loader)
        x = x.to(self.device)
        y = y.to(self.device)
        if self.ridge_head is None:
            self.ridge_head = RidgeAccumulator(x.shape[1], self.num_classes).to(self.device)
        ridge = self.ridge_head.select_ridge_stratified_accuracy(x, y, self.ridge_candidates, n_splits=self.ridge_splits)
        self.ridge_head.update(x, y)
        self.ridge_head.solve(ridge)
        trainer.logger.info("task=%s layup_ridge=%.3g samples=%d dim=%d", task.name, ridge, y.numel(), x.shape[1])
        trainer.log_metrics({"train/layup_ridge": ridge, "train/task_index": float(_task_id(task))})
        return True

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.ridge_head is None:
            raise RuntimeError("LayUP has no ridge classifier; train at least one task first.")
        x = batch["x"].to(self.device)
        z = self.feature_extractor(x)
        logits = z.matmul(self.ridge_head.weight.to(device=z.device, dtype=z.dtype).t())
        return {"logits": logits, "features": z}


class OnlineTruncatedSVDSolver(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, rank: int = 20000, ridge: float = 0.0, truncate_percent: float = 25.0) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.truncate_percent = float(truncate_percent)
        self.register_buffer("u", torch.empty(self.feature_dim, 0))
        self.register_buffer("s", torch.empty(0))
        self.register_buffer("cov_hy", torch.zeros(self.feature_dim, self.num_classes))
        self.num_samples = 0

    def update(self, h: torch.Tensor, y: torch.Tensor) -> None:
        h = h.detach().float().to(self.cov_hy.device)
        y = y.long().to(self.cov_hy.device)
        self.cov_hy += h.t().matmul(_one_hot(y, self.num_classes, dtype=h.dtype))
        self._update_svd(h)
        self.num_samples += int(h.shape[0])

    def _update_svd(self, h: torch.Tensor) -> None:
        columns = h.t()
        if self.u.numel():
            summary = self.u * self.s.reshape(1, -1)
            columns = torch.cat([summary.to(columns.device), columns], dim=1)
        max_rank = max(1, min(self.rank, columns.shape[0], columns.shape[1]))
        keep_by_samples = int(math.ceil(columns.shape[1] * max(0.0, 1.0 - self.truncate_percent / 100.0)))
        keep = max(1, min(max_rank, keep_by_samples))
        u, s, _vh = torch.linalg.svd(columns, full_matrices=False)
        self.u = u[:, :keep].detach()
        self.s = s[:keep].detach()

    def weight(self) -> torch.Tensor:
        if self.u.numel() == 0:
            return torch.zeros(self.num_classes, self.feature_dim, device=self.cov_hy.device)
        ut_cov = self.u.t().matmul(self.cov_hy)
        denom = self.s.square().unsqueeze(1) + float(self.ridge)
        return self.u.matmul(ut_cov / denom.clamp_min(1e-12)).t()


@register_method("loranpac")
@register_method("lo_ranpac")
class LoRanPACMethod(FrozenFeatureMethod):
    """LoRanPAC low-rank random-feature ridge solver."""

    def __init__(
        self,
        E: int = 100000,
        rank: int = 20000,
        truncate_percent: float = 25.0,
        ridge: float = 0.0,
        use_RE: bool = True,
        use_relu: bool = True,
        coslinear: bool = False,
        tsvd_batch_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(freeze_backbone=True, **kwargs)
        self.E = int(E)
        self.use_RE = bool(use_RE)
        self.coslinear = bool(coslinear)
        self.tsvd_batch_size = int(tsvd_batch_size)
        feature_dim = int(self.detector.feature_dim)
        proj_dim = self.E if self.use_RE else feature_dim
        self.projector = RandomProjection(feature_dim, self.E if self.use_RE else 0, use_relu=use_relu)
        self.solver = OnlineTruncatedSVDSolver(proj_dim, self.num_classes, rank=rank, ridge=ridge, truncate_percent=truncate_percent)

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        features, labels = self.collect_features(train_loader)
        for start in range(0, features.shape[0], max(self.tsvd_batch_size, 1)):
            stop = start + max(self.tsvd_batch_size, 1)
            h = self.projector(features[start:stop].to(self.device))
            self.solver.update(h, labels[start:stop].to(self.device))
        dim = self.projector.out_dim if self.use_RE else int(self.detector.feature_dim)
        trainer.logger.info("task=%s loranpac_rank=%d samples=%d dim=%d", task.name, self.solver.s.numel(), labels.numel(), dim)
        trainer.log_metrics({"train/loranpac_rank": float(self.solver.s.numel()), "train/task_index": float(_task_id(task))})
        return True

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.extract_features(x)
        h = self.projector(z)
        w = self.solver.weight().to(device=h.device, dtype=h.dtype)
        if self.coslinear:
            logits = F.normalize(h, dim=-1).matmul(F.normalize(w, dim=-1).t())
        else:
            logits = h.matmul(w.t())
        return {"logits": logits, "features": z}


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
            centers = F.normalize(torch.as_tensor(centers_np, dtype=torch.float32), dim=-1)
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
            distances.append(torch.cdist(z_norm, centers).min(dim=1).values)
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


@register_method("cp_prompt")
@register_method("cp-prompt")
@register_method("cpprompt")
class CPPromptMethod(DomainRoutedFeatureMethod):
    """CP-Prompt composition-style common and personalized prompt reproduction."""

    def __init__(self, prompt_dim: int | None = None, common_prompt_lr_scale: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        dim = int(prompt_dim or self.detector.feature_dim)
        if dim != int(self.detector.feature_dim):
            raise ValueError("Feature-space CP-Prompt requires prompt_dim == detector.feature_dim.")
        self.common_prompt = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.common_prompt, std=0.02)
        self.personal_prompts = nn.ParameterDict()
        self.common_prompt_lr_scale = float(common_prompt_lr_scale)

    def _ensure_task_modules(self, key: str) -> None:
        super()._ensure_task_modules(key)
        if key not in self.personal_prompts:
            prompt = torch.empty(int(self.detector.feature_dim))
            nn.init.normal_(prompt, std=0.02)
            self.personal_prompts[key] = nn.Parameter(prompt)

    def _freeze_except_current(self) -> None:
        super()._freeze_except_current()
        for key, prompt in self.personal_prompts.items():
            prompt.requires_grad_(key == self.current_key)
        self.common_prompt.requires_grad_(True)

    def _task_logits(self, z: torch.Tensor, key: str) -> torch.Tensor:
        composed = z + self.common_prompt.to(z.dtype) + self.personal_prompts[key].to(z.dtype)
        return self.classifiers[key](self.adapters[key](composed))


class SOYOSelector(nn.Module):
    def __init__(self, in_dim: int, max_tasks: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(max_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.float())


@register_method("soyo")
class SOYOMethod(DomainRoutedFeatureMethod):
    """SOYO parameter-isolation plus learned domain selector."""

    def __init__(
        self,
        total_sessions: int = 5,
        gmm_components: int = 2,
        soyo_epoch: int = 30,
        soyo_lr: float = 0.1,
        soyo_weight_decay: float = 2e-4,
        resample_per_domain: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.total_sessions = int(total_sessions)
        self.gmm_components = int(gmm_components)
        self.soyo_epoch = int(soyo_epoch)
        self.soyo_lr = float(soyo_lr)
        self.soyo_weight_decay = float(soyo_weight_decay)
        self.resample_per_domain = int(resample_per_domain)
        self.selector = SOYOSelector(int(self.detector.feature_dim), self.total_sessions)
        self.gmms: dict[str, GaussianMixture] = {}

    def _route(self, z: torch.Tensor) -> torch.Tensor:
        keys = list(self.adapters.keys())
        if not keys:
            return torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        logits = self.selector(z)
        return logits[:, : len(keys)].argmax(dim=1)

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is None:
            return
        features, _labels = self.collect_features(train_loader)
        arr = features.float().numpy()
        if len(arr):
            n_components = min(max(1, self.gmm_components), len(arr))
            self.gmms[self.current_key] = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0).fit(arr)
        self._train_selector(features.to(self.device), self.current_key)
        self._store_centers(self.current_key, train_loader)
        self._freeze_except_current()

    def _train_selector(self, current_features: torch.Tensor, current_key: str) -> None:
        keys = list(self.adapters.keys())
        if not keys or current_features.numel() == 0:
            return
        features = []
        labels = []
        key_to_id = {key: i for i, key in enumerate(keys)}
        for key, gmm in self.gmms.items():
            if key == current_key:
                continue
            count = max(self.resample_per_domain, 1)
            samples, _ = gmm.sample(count)
            features.append(torch.as_tensor(samples, dtype=torch.float32, device=self.device))
            labels.append(torch.full((count,), key_to_id[key], dtype=torch.long, device=self.device))
        features.append(current_features.float())
        labels.append(torch.full((current_features.shape[0],), key_to_id[current_key], dtype=torch.long, device=self.device))
        x = torch.cat(features, dim=0)
        y = torch.cat(labels, dim=0)
        optimizer = torch.optim.SGD(self.selector.parameters(), lr=self.soyo_lr, momentum=0.9, weight_decay=self.soyo_weight_decay)
        self.selector.train()
        for _epoch in range(max(self.soyo_epoch, 1)):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.selector(x)[:, : len(keys)], y)
            loss.backward()
            optimizer.step()


class CosineLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))


class DCEExpert(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, max(dim // 2, 1)))
        self.head = CosineLinear(max(dim // 2, 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.float()))


@register_method("dce")
class DCEMethod(FrozenFeatureMethod):
    """Dual-Balance Collaborative Experts for imbalanced DIL."""

    def __init__(
        self,
        total_sessions: int = 7,
        bal_epoch: int = 10,
        selector_epoch: int = 10,
        selector_lr: float = 0.01,
        num_sampled_pcls: int = 256,
        use_sm: bool = False,
        margin_sample_num: int = 10,
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
        super().__init__(freeze_backbone=True, **kwargs)
        self.total_sessions = int(total_sessions)
        self.bal_epoch = int(bal_epoch)
        self.selector_epoch = int(selector_epoch)
        self.selector_lr = float(selector_lr)
        self.num_sampled_pcls = int(num_sampled_pcls)
        self.use_sm = bool(use_sm)
        self.margin_sample_num = int(margin_sample_num)
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
        self.naive = nn.ModuleDict()
        self.balanced = nn.ModuleDict()
        self.reverse = nn.ModuleDict()
        self.selector = nn.Linear(int(self.detector.feature_dim), 3 * self.total_sessions)
        self.stats: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.current_key = "task0"

    def _ensure_experts(self, key: str) -> None:
        if key in self.naive:
            return
        dim = int(self.detector.feature_dim)
        self.naive[key] = DCEExpert(dim, self.num_classes)
        self.balanced[key] = DCEExpert(dim, self.num_classes)
        self.reverse[key] = DCEExpert(dim, self.num_classes)

    def _freeze_except_current(self) -> None:
        for pool in [self.naive, self.balanced, self.reverse]:
            for key, module in pool.items():
                for p in module.parameters():
                    p.requires_grad_(key == self.current_key)
        for p in self.selector.parameters():
            p.requires_grad_(False)

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.current_task_id = _task_id(task)
        self.current_key = f"task{self.current_task_id}"
        self._ensure_experts(self.current_key)
        self._freeze_except_current()

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        cfg = _official_optimizer_cfg(
            optimizer_cfg,
            self.current_task_id or 0,
            init_lr=self.init_lr,
            lr=self.lr,
            lrate=self.lrate,
            init_weight_decay=self.init_weight_decay,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
        )
        cfg.setdefault("lr", 0.01)
        cfg.setdefault("weight_decay", 5e-4)
        return build_optimizer(self.parameters(), cfg)

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        task_id = _task_id(task)
        epochs = _official_task_epochs(trainer, task_id, self.init_epoch, self.epochs)
        optimizer = self.configure_optimizer(trainer.optimizer_cfg)
        scheduler = _official_scheduler(
            optimizer,
            task_id,
            init_milestones=self.init_milestones,
            milestones=self.milestones,
            init_lr_decay=self.init_lr_decay,
            lr_decay=self.lr_decay,
            lrate_decay=self.lrate_decay,
        )
        _run_minibatch_loop(self, trainer, task, train_loader, optimizer, epochs, scheduler)
        return True

    def _counts(self, y: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(y.long(), minlength=self.num_classes).float().to(y.device)
        return counts.clamp_min(1.0)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        z = self.extract_features(batch["x"])
        y = _local_targets(batch["y"], self.num_classes)
        counts = self._counts(y)
        logits_n = self.naive[self.current_key](z)
        logits_b = self.balanced[self.current_key](z)
        logits_r = self.reverse[self.current_key](z)
        log_counts = counts.log().view(1, -1)
        loss_n = F.cross_entropy(logits_n, y)
        loss_b = F.cross_entropy(logits_b + log_counts, y)
        loss_r = F.cross_entropy(logits_r + 2.0 * log_counts, y)
        loss = loss_n + loss_b + loss_r
        return {"loss": loss, "ce": loss_n.detach(), "balanced_ce": loss_b.detach(), "reverse_ce": loss_r.detach()}

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is None:
            return
        features, labels = self.collect_features(train_loader)
        self._update_stats(self.current_key, features, labels)
        self._train_selector_from_stats()
        self._freeze_except_current()

    def _update_stats(self, key: str, features: torch.Tensor, labels: torch.Tensor) -> None:
        for cls in range(self.num_classes):
            z = features[labels == cls].float()
            if z.numel() == 0:
                continue
            mean = z.mean(dim=0)
            if z.shape[0] >= max(self.margin_sample_num, 2):
                centered = z - mean
                cov = centered.t().matmul(centered) / max(z.shape[0] - 1, 1)
            else:
                cov = torch.eye(z.shape[1]) * 1e-3
            cov = cov + torch.eye(cov.shape[0]) * 1e-4
            self.stats[(key, cls)] = (mean.detach(), cov.detach(), int(z.shape[0]))

    def _sample_stats(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        xs = []
        ys = []
        for (_key, cls), (mean, cov, _count) in self.stats.items():
            try:
                dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
                sample = dist.sample((self.num_sampled_pcls,))
            except Exception:
                sample = mean.unsqueeze(0).repeat(self.num_sampled_pcls, 1)
            xs.append(sample)
            ys.append(torch.full((sample.shape[0],), int(cls), dtype=torch.long))
        if not xs:
            return None
        return torch.cat(xs, dim=0).to(self.device), torch.cat(ys, dim=0).to(self.device)

    def _train_selector_from_stats(self) -> None:
        payload = self._sample_stats()
        if payload is None:
            return
        x, y = payload
        keys = list(self.naive.keys())
        if not keys:
            return
        for p in self.selector.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(self.selector.parameters(), lr=self.selector_lr, momentum=0.9, weight_decay=2e-4)
        self.selector.train()
        for _epoch in range(max(self.selector_epoch, 1)):
            expert_logits = self._all_expert_logits(x, keys).detach()
            weights = self.selector(x)[:, : 3 * len(keys)]
            weights = F.softmax(weights, dim=1) if self.use_sm else weights
            logits = torch.einsum("be,bec->bc", weights, expert_logits)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for p in self.selector.parameters():
            p.requires_grad_(False)

    def _all_expert_logits(self, z: torch.Tensor, keys: list[str]) -> torch.Tensor:
        logits = []
        for key in keys:
            logits.extend([self.naive[key](z), self.balanced[key](z), self.reverse[key](z)])
        return torch.stack(logits, dim=1)

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.extract_features(x)
        keys = list(self.naive.keys())
        expert_logits = self._all_expert_logits(z, keys)
        weights = self.selector(z)[:, : 3 * len(keys)]
        weights = F.softmax(weights, dim=1) if self.use_sm else weights
        logits = torch.einsum("be,bec->bc", weights, expert_logits)
        return {"logits": logits, "features": z, "expert_weights": weights.detach()}


@register_method("duct")
class DUCTMethod(ContinualMethod):
    """DUCT representation consolidation with a framework-compatible head."""

    def __init__(
        self,
        merge_scalar: float = 0.5,
        retrain_epochs: int = 5,
        epc_re: int | None = None,
        lr_re: float = 1e-3,
        head_merge_ratio: float = 0.5,
        bcb_lr_scale: float = 0.01,
        ot_reg: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.merge_scalar = float(merge_scalar)
        self.retrain_epochs = int(retrain_epochs if epc_re is None else epc_re)
        self.lr_re = float(lr_re)
        self.head_merge_ratio = float(head_merge_ratio)
        self.bcb_lr_scale = float(bcb_lr_scale)
        self.ot_reg = float(ot_reg)
        self._init_backbone_state = _clone_state(self.detector.backbone)
        self._merged_delta: dict[str, torch.Tensor] = {
            key: torch.zeros_like(value) for key, value in self._init_backbone_state.items()
        }
        self._previous_head: dict[str, torch.Tensor] | None = None

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        self.train()
        optimizer = self.configure_optimizer(trainer.optimizer_cfg)
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
                totals["ce"] = totals.get("ce", 0.0) + float(out["ce"].detach().cpu())
                n += 1
            if totals:
                trainer.logger.info("task=%s epoch=%d/%d ce=%.4f", task.name, epoch + 1, trainer.max_epochs, totals["ce"] / max(n, 1))
        self._merge_backbone()
        self._retrain_head(trainer, train_loader)
        return True

    def _merge_backbone(self) -> None:
        current = _clone_state(self.detector.backbone)
        delta = _state_delta(current, self._init_backbone_state)
        for key, value in delta.items():
            self._merged_delta[key] = self._merged_delta[key] + self.merge_scalar * value
        merged = {
            key: self._init_backbone_state[key] + self._merged_delta.get(key, torch.zeros_like(self._init_backbone_state[key]))
            for key in self._init_backbone_state
        }
        _load_float_state(self.detector.backbone, merged)

    def _retrain_head(self, trainer: Any, train_loader: Any) -> None:
        if self.retrain_epochs <= 0:
            return
        for p in self.detector.backbone.parameters():
            p.requires_grad_(False)
        for p in self.detector.head.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(self.detector.head.parameters(), lr=self.lr_re, momentum=0.9)
        self.train()
        for _epoch in range(self.retrain_epochs):
            for batch in train_loader:
                batch = batch_to_device(batch, self.device)
                out = self.detector(batch["x"])
                loss = F.cross_entropy(out["logits"], _local_targets(batch["y"], self.num_classes))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                trainer.advance_step()
        if self._previous_head is not None:
            state = self.detector.head.state_dict()
            for key, value in state.items():
                if key in self._previous_head and torch.is_floating_point(value) and value.shape == self._previous_head[key].shape:
                    state[key] = self.head_merge_ratio * self._previous_head[key].to(value.device) + (1.0 - self.head_merge_ratio) * value
            self.detector.head.load_state_dict(state, strict=False)
        self._previous_head = {k: v.detach().cpu().clone() for k, v in self.detector.head.state_dict().items() if torch.is_floating_point(v)}
        for p in self.detector.backbone.parameters():
            p.requires_grad_(True)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        out = self.detector(batch["x"])
        ce = F.cross_entropy(out["logits"], _local_targets(batch["y"], self.num_classes))
        return {"loss": ce, "ce": ce.detach()}
