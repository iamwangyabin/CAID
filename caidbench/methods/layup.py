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
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            features.append(self.extract_features(batch["x"]).detach().cpu())
            labels.append(_local_targets(batch["y"].detach().cpu(), self.num_classes))
        if was_training:
            self.train()
        if not features:
            return torch.empty(0, int(self.detector.feature_dim)), torch.empty(0, dtype=torch.long)
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _torch_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower()
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    if key in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"Unsupported ridge dtype: {name!r}. Use 'float32' or 'float64'.")


class RidgeAccumulator(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dtype: torch.dtype = torch.float32) -> None:
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


class CosineLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))


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
        use_test_transform_for_ridge: bool = True,
        ridge_dtype: str | torch.dtype = "float32",
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
        self.ridge_candidates = list(ridge_candidates or [*np.logspace(-4, 3, 15).tolist(), 1e-8])
        self.ridge_splits = int(ridge_splits)
        self.use_test_transform_for_ridge = bool(use_test_transform_for_ridge)
        self.ridge_dtype = _torch_dtype(ridge_dtype)
        self.ridge_head: RidgeAccumulator | None = None

    def _default_vit_layers(self, k: int) -> list[str]:
        backbone = getattr(self.detector, "backbone", None)
        model = getattr(backbone, "model", None)
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            return []
        n_blocks = len(blocks)
        return [f"backbone.model.blocks.{n_blocks - 1 - i}" for i in range(min(max(k, 0), n_blocks))]

    def _build_ridge_loader(self, trainer: Any, task: Any, fallback_loader: Any) -> Any:
        """Match official LayUP: collect current train activations with test transforms."""
        if not self.use_test_transform_for_ridge:
            return fallback_loader
        scenario = getattr(trainer, "scenario", None)
        if scenario is None or not hasattr(scenario, "source"):
            return fallback_loader

        task_index = None
        for idx, spec in enumerate(getattr(scenario, "tasks", [])):
            if spec is task or (
                getattr(spec, "task_id", None) == getattr(task, "task_id", None)
                and getattr(spec, "name", None) == getattr(task, "name", None)
            ):
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
            shuffle=False,
            num_workers=int(getattr(trainer, "num_workers", 0)),
            drop_last=False,
        )

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
            for _batch_idx, batch in iter_limited_train_batches(trainer, train_loader):
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
            trainer.log_train_metrics(
                {
                    "layup_fsa_loss": total_loss / max(batches, 1),
                    "layup_fsa_acc": train_acc,
                    "layup_fsa_val_acc": val_acc,
                },
                task=task,
                epoch=epoch + 1,
                epochs=max(self.finetune_epochs, 1),
                optimizer=optimizer,
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
        ridge_loader = self._build_ridge_loader(trainer, task, train_loader)
        x, y = self.collect_features(ridge_loader)
        x = x.to(self.device)
        y = y.to(self.device)
        if self.ridge_head is None:
            self.ridge_head = RidgeAccumulator(x.shape[1], self.num_classes, dtype=self.ridge_dtype).to(self.device)
        ridge = self.ridge_head.select_ridge_stratified_accuracy(x, y, self.ridge_candidates, n_splits=self.ridge_splits)
        self.ridge_head.update(x, y)
        self.ridge_head.solve(ridge)
        trainer.log_train_metrics(
            {
                "layup_ridge": ridge,
                "layup_samples": float(y.numel()),
                "layup_dim": float(x.shape[1]),
            },
            task=task,
            phase="ridge",
        )
        return True

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.ridge_head is None:
            raise RuntimeError("LayUP has no ridge classifier; train at least one task first.")
        x = batch["x"].to(self.device)
        z = self.feature_extractor(x)
        logits = z.matmul(self.ridge_head.weight.to(device=z.device, dtype=z.dtype).t())
        return {"logits": logits, "features": z}
