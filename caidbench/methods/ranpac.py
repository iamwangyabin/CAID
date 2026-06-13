from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..data.loader import build_dataloader
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, freeze_module, iter_limited_train_batches


def _local_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = y.long()
    if y.numel() and (int(y.min()) < 0):
        raise ValueError("Labels must be non-negative for domain-incremental methods.")
    return torch.remainder(y, int(num_classes))


def _task_id(task: Any) -> int:
    return int(getattr(task, "task_id", task if isinstance(task, int) else 0))


def _one_hot(y: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return F.one_hot(y.long(), num_classes=int(num_classes)).to(dtype=dtype)


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
        for _batch_idx, batch in iter_limited_train_batches(self, loader):
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


class RanPACCosineLinear(nn.Module):
    """Official RanPAC CosineLinear head used during first-task PETL tuning."""

    def __init__(self, in_features: int, out_features: int, sigma: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.empty(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-bound, bound)
        if self.sigma is not None:
            self.sigma.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        return self.sigma * out if self.sigma is not None else out


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
        adapter_dropout: float = 0.1,
        adapter_scalar: float = 0.1,
        use_test_transform_for_ridge: bool = True,
        use_official_cosine_head: bool = True,
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
        self.use_official_cosine_head = bool(use_official_cosine_head)
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
        if self.use_official_cosine_head and (self.model_name != "ncm" or self.tuned_epoch > 0):
            self.detector.head = RanPACCosineLinear(int(self.detector.feature_dim), self.num_classes)
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
            for _batch_idx, batch in iter_limited_train_batches(trainer, train_loader):
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
            trainer.log_train_metrics(
                {
                    "ranpac_tune_loss": avg_loss,
                    "ranpac_tune_acc": train_acc,
                    "ranpac_tune_eval_acc": tune_acc,
                },
                task=task,
                epoch=epoch + 1,
                epochs=self.tuned_epoch,
                optimizer=optimizer,
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
        trainer.log_train_metrics(
            {
                "ranpac_ridge": ridge,
                "ranpac_samples": float(labels.numel()),
                "ranpac_dim": float(h.shape[1]),
            },
            task=task,
            phase="ridge",
        )
        return True

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        z = self.extract_features(x)
        h = self.projector(z)
        logits = h.matmul(self.ridge_head.weight.to(device=h.device, dtype=h.dtype).t())
        return {"logits": logits, "features": z}
