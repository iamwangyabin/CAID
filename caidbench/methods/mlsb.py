from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..data.loader import build_dataloader
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, freeze_module, iter_limited_train_batches


def _storage_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower()
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported MLSB storage_dtype={name!r}. Use float16, bfloat16, or float32.")


def _binary_targets(y: torch.Tensor, real_label: int = 0, fake_label: int = 1) -> torch.Tensor:
    y = y.long()
    if y.numel() and int(y.min()) < 0:
        raise ValueError("MLSB labels must be non-negative.")
    unique = set(int(v) for v in y.detach().cpu().unique().tolist())
    configured = {int(real_label), int(fake_label)}
    if unique.issubset(configured):
        return (y == int(fake_label)).long()
    return torch.remainder(y, 2).long()


class HookedFeatureExtractor:
    """Extract RINE-style intermediate activations plus optional final features."""

    def __init__(
        self,
        detector: nn.Module,
        layer_names: Sequence[str] | None = None,
        *,
        include_final_feature: bool = False,
        token_pool: str = "cls",
    ) -> None:
        self.detector = detector
        self.layer_names = [str(name) for name in (layer_names or [])]
        self.include_final_feature = bool(include_final_feature)
        self.token_pool = str(token_pool).lower()

    @staticmethod
    def _pool(out: Any, token_pool: str, batch_size: int) -> torch.Tensor:
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"Hooked layer returned {type(out)!r}, expected tensor.")
        if out.ndim == 4:
            return F.adaptive_avg_pool2d(out.float(), 1).flatten(1)
        if out.ndim == 3:
            z = out.float()
            if z.shape[0] == batch_size:
                return z.mean(dim=1) if token_pool == "mean" else z[:, 0]
            if z.shape[1] == batch_size:
                return z.mean(dim=0) if token_pool == "mean" else z[0]
            return z.mean(dim=1) if token_pool == "mean" else z[:, 0]
        return out.reshape(out.shape[0], -1).float()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations: dict[str, torch.Tensor] = {}
        handles: list[Any] = []
        for name in self.layer_names:
            module = self.detector.get_submodule(name)
            handles.append(module.register_forward_hook(lambda _m, _i, out, key=name: activations.setdefault(key, out)))
        try:
            final = self.detector.extract_features(x)
        finally:
            for handle in handles:
                handle.remove()

        parts = [self._pool(activations[name], self.token_pool, int(x.shape[0])) for name in self.layer_names]
        if self.include_final_feature or not parts:
            parts.append(final.float())
        return parts

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward(x)


@register_method("sparse_boundary")
@register_method("frozen_boundary")
@register_method("mlsb")
class MLSBMethod(ContinualMethod):
    """Frozen multi-layer sparse boundary baseline.

    MLSB follows RINE's observation that CLIP/ViT intermediate blocks are useful
    for synthetic image detection, but keeps the continual learner sample-free:
    after each task it stores only a real-vs-fake hyperplane per selected layer.
    """

    _DYNAMIC_BUFFER_KEYS = (
        "boundary_weight",
        "boundary_indices",
        "boundary_values",
        "boundary_bias",
        "boundary_task_ids",
        "layer_dims",
    )

    def __init__(
        self,
        detector_cfg: dict[str, Any] | None = None,
        num_classes: int = 2,
        *,
        freeze_backbone: bool = True,
        feature_mode: str = "auto_hooks",
        hook_pattern: str = "auto",
        layer_names: Sequence[str] | None = None,
        k_layers: int = 0,
        include_final_feature: bool = False,
        token_pool: str = "cls",
        normalize_layer_features: bool = True,
        normalize_concat: bool = False,
        top_k: int = 128,
        storage_dtype: str | torch.dtype = "float16",
        use_test_transform_for_stats: bool = True,
        real_label: int = 0,
        fake_label: int = 1,
        min_samples_per_class: int = 1,
        **kwargs: Any,
    ) -> None:
        if int(num_classes) != 2:
            raise ValueError("MLSB is a binary real/fake baseline and requires num_classes=2.")
        super().__init__(detector_cfg=detector_cfg, num_classes=num_classes, **kwargs)
        self.freeze_backbone = bool(freeze_backbone)
        self.feature_mode = str(feature_mode).lower()
        self.hook_pattern = str(hook_pattern)
        self.k_layers = int(k_layers)
        self.include_final_feature = bool(include_final_feature)
        self.token_pool = str(token_pool).lower()
        self.normalize_layer_features = bool(normalize_layer_features)
        self.normalize_concat = bool(normalize_concat)
        self.top_k = int(top_k)
        self.use_test_transform_for_stats = bool(use_test_transform_for_stats)
        self.real_label = int(real_label)
        self.fake_label = int(fake_label)
        self.min_samples_per_class = int(min_samples_per_class)
        self.storage_dtype = _storage_dtype(storage_dtype)
        if self.freeze_backbone:
            freeze_module(self.detector)

        resolved_layers = self._resolve_layer_names(layer_names)
        self.feature_extractor = HookedFeatureExtractor(
            self.detector,
            resolved_layers,
            include_final_feature=self.include_final_feature,
            token_pool=self.token_pool,
        )

        self.register_buffer("boundary_weight", torch.empty(0, 0, dtype=self.storage_dtype))
        self.register_buffer("boundary_indices", torch.empty(0, 0, dtype=torch.int32))
        self.register_buffer("boundary_values", torch.empty(0, 0, dtype=self.storage_dtype))
        self.register_buffer("boundary_bias", torch.empty(0, 0, dtype=self.storage_dtype))
        self.register_buffer("boundary_task_ids", torch.empty(0, dtype=torch.long))
        self.register_buffer("layer_dims", torch.empty(0, dtype=torch.long))

    def _resolve_layer_names(self, layer_names: Sequence[str] | None) -> list[str]:
        if layer_names:
            return [str(name) for name in layer_names]
        if self.feature_mode in {"none", "final", "final_only"}:
            return []

        modules = [(name, module) for name, module in self.detector.named_modules() if name]
        patterns = [self.hook_pattern]
        if self.hook_pattern.lower() == "auto":
            patterns = ["ln_2", "norm2", "layer_norm2"]

        for pattern in patterns:
            matches = [name for name, module in modules if pattern in name and not list(module.children())]
            if matches:
                if self.k_layers > 0:
                    return matches[-self.k_layers :]
                return matches
        return []

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_backbone:
            self.detector.eval()
        return self

    def _build_stats_loader(self, trainer: Any, task: Any, fallback_loader: Any) -> Any:
        if not self.use_test_transform_for_stats:
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

        indices = getattr(scenario, "_split_indices", {}).get((task_index, "train"))
        if indices is None:
            return fallback_loader
        dataset = scenario.source.make_dataset(
            indices,
            transform_cfg=scenario._transform_for_split("test"),
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

    def _normalize_layers(self, layers: list[torch.Tensor]) -> list[torch.Tensor]:
        out = [z.float() for z in layers]
        if self.normalize_layer_features:
            out = [F.normalize(z, dim=-1) for z in out]
        if self.normalize_concat and out:
            dims = [z.shape[1] for z in out]
            merged = F.normalize(torch.cat(out, dim=1), dim=-1)
            out = list(merged.split(dims, dim=1))
        return out

    @torch.no_grad()
    def _extract_feature_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        layers = self.feature_extractor(x.to(self.device))
        return self._normalize_layers(layers)

    def _check_layer_dims(self, weights: Sequence[torch.Tensor]) -> None:
        dims = torch.tensor([int(w.numel()) for w in weights], dtype=torch.long, device=self.device)
        if self.layer_dims.numel() == 0:
            self.layer_dims = dims
            return
        if self.layer_dims.numel() != dims.numel() or not torch.equal(self.layer_dims.to(dims.device), dims):
            raise RuntimeError(
                f"MLSB feature dimensions changed from {self.layer_dims.detach().cpu().tolist()} "
                f"to {dims.detach().cpu().tolist()}."
            )

    def _append_boundary(self, weights: Sequence[torch.Tensor], biases: torch.Tensor, task_id: int) -> None:
        if not weights:
            raise RuntimeError("MLSB cannot append an empty boundary.")
        weights = [w.detach().float().flatten().to(self.device) for w in weights]
        biases = biases.detach().float().flatten().to(self.device)
        if biases.numel() != len(weights):
            raise ValueError(f"Expected one bias per layer, got {biases.numel()} biases for {len(weights)} layers.")
        self._check_layer_dims(weights)

        if self.top_k <= 0:
            dense = torch.cat(weights, dim=0).to(dtype=self.storage_dtype).unsqueeze(0)
            if self.boundary_weight.shape[0] == 0:
                self.boundary_weight = dense
            else:
                self.boundary_weight = torch.cat([self.boundary_weight.to(dense.device), dense], dim=0)
        else:
            indices: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            offset = 0
            for w in weights:
                k = min(int(self.top_k), int(w.numel()))
                idx = torch.topk(w.abs(), k=k, largest=True).indices.sort().values
                indices.append((idx + offset).to(dtype=torch.int32))
                values.append(w[idx].to(dtype=self.storage_dtype))
                offset += int(w.numel())
            row_indices = torch.cat(indices, dim=0).unsqueeze(0)
            row_values = torch.cat(values, dim=0).unsqueeze(0)
            if self.boundary_indices.shape[0] == 0:
                self.boundary_indices = row_indices
                self.boundary_values = row_values
            else:
                self.boundary_indices = torch.cat([self.boundary_indices.to(row_indices.device), row_indices], dim=0)
                self.boundary_values = torch.cat([self.boundary_values.to(row_values.device), row_values], dim=0)

        row_bias = biases.to(dtype=self.storage_dtype).unsqueeze(0)
        if self.boundary_bias.shape[0] == 0:
            self.boundary_bias = row_bias
        else:
            self.boundary_bias = torch.cat([self.boundary_bias.to(row_bias.device), row_bias], dim=0)
        row_task = torch.tensor([int(task_id)], dtype=torch.long, device=self.device)
        if self.boundary_task_ids.shape[0] == 0:
            self.boundary_task_ids = row_task
        else:
            self.boundary_task_ids = torch.cat([self.boundary_task_ids.to(row_task.device), row_task], dim=0)

    def _compute_boundary(
        self,
        sum_real: Sequence[torch.Tensor],
        sum_fake: Sequence[torch.Tensor],
        count_real: int,
        count_fake: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if count_real < self.min_samples_per_class or count_fake < self.min_samples_per_class:
            raise RuntimeError(
                f"MLSB needs at least {self.min_samples_per_class} real and fake samples per task; "
                f"got real={count_real}, fake={count_fake}."
            )
        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for r_sum, f_sum in zip(sum_real, sum_fake):
            mu_real = r_sum.float() / float(count_real)
            mu_fake = f_sum.float() / float(count_fake)
            w = mu_fake - mu_real
            b = -0.5 * torch.dot(mu_fake + mu_real, w)
            weights.append(w)
            biases.append(b)
        return weights, torch.stack(biases)

    def _memory_bytes(self) -> int:
        tensors = [self.boundary_weight, self.boundary_indices, self.boundary_values, self.boundary_bias, self.boundary_task_ids, self.layer_dims]
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        del val_loader
        stats_loader = self._build_stats_loader(trainer, task, train_loader)
        was_training = self.training
        self.eval()
        sum_real: list[torch.Tensor] | None = None
        sum_fake: list[torch.Tensor] | None = None
        count_real = 0
        count_fake = 0
        for _batch_idx, batch in iter_limited_train_batches(trainer, stats_loader):
            batch = batch_to_device(batch, self.device)
            layers = self._extract_feature_layers(batch["x"])
            y = _binary_targets(batch["y"], self.real_label, self.fake_label).to(self.device)
            if sum_real is None or sum_fake is None:
                sum_real = [torch.zeros(z.shape[1], device=self.device, dtype=torch.float32) for z in layers]
                sum_fake = [torch.zeros(z.shape[1], device=self.device, dtype=torch.float32) for z in layers]
            real_mask = y == 0
            fake_mask = y == 1
            count_real += int(real_mask.sum().item())
            count_fake += int(fake_mask.sum().item())
            for idx, z in enumerate(layers):
                if real_mask.any():
                    sum_real[idx] += z[real_mask].sum(dim=0)
                if fake_mask.any():
                    sum_fake[idx] += z[fake_mask].sum(dim=0)

        if sum_real is None or sum_fake is None:
            raise RuntimeError("MLSB received no batches while fitting task boundary.")
        weights, biases = self._compute_boundary(sum_real, sum_fake, count_real, count_fake)
        task_id = int(getattr(task, "task_id", len(self.boundary_task_ids)))
        self._append_boundary(weights, biases, task_id)
        if was_training:
            self.train()
        trainer.log_train_metrics(
            {
                "mlsb_samples": float(count_real + count_fake),
                "mlsb_layers": float(len(weights)),
                "mlsb_dim": float(sum(int(w.numel()) for w in weights)),
                "mlsb_memory_bytes": float(self._memory_bytes()),
                "mlsb_top_k": float(self.top_k),
            },
            task=task,
            phase="boundary",
        )
        return True

    def _score_layers(self, layers: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.boundary_task_ids.numel() == 0:
            raise RuntimeError("MLSB has no task boundaries; fit at least one task first.")
        layers = [z.float().to(self.device) for z in layers]
        merged = torch.cat(layers, dim=1)
        layer_count = max(int(self.layer_dims.numel()), 1)
        if self.top_k <= 0:
            weight = self.boundary_weight.to(device=merged.device, dtype=merged.dtype)
            scores = merged.matmul(weight.t())
        else:
            indices = self.boundary_indices.to(device=merged.device, dtype=torch.long)
            values = self.boundary_values.to(device=merged.device, dtype=merged.dtype)
            scores = (merged[:, indices] * values.unsqueeze(0)).sum(dim=-1)
        bias = self.boundary_bias.to(device=merged.device, dtype=merged.dtype).sum(dim=1)
        return (scores + bias.unsqueeze(0)) / float(layer_count)

    @torch.no_grad()
    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        layers = self._extract_feature_layers(x)
        scores = self._score_layers(layers)
        final_score, selection = scores.max(dim=1)
        logits = torch.stack([-final_score, final_score], dim=1)
        features = torch.cat(layers, dim=1)
        return {"logits": logits, "features": features, "task_selection": selection.detach()}

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        del task
        batch = batch_to_device(batch, self.device)
        out = self.predict(batch)
        y = _binary_targets(batch["y"], self.real_label, self.fake_label).to(out["logits"].device)
        loss = F.cross_entropy(out["logits"], y)
        return {"loss": loss, "ce": loss.detach(), "logits": out["logits"].detach()}

    def _prepare_dynamic_buffers(self, state: Mapping[str, Any]) -> None:
        for key in self._DYNAMIC_BUFFER_KEYS:
            value = state.get(key)
            if torch.is_tensor(value):
                setattr(self, key, value.detach().clone())

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):  # type: ignore[override]
        self._prepare_dynamic_buffers(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def load_checkpoint_state_dict(self, state: Mapping[str, Any]):
        self._prepare_dynamic_buffers(state)
        return super().load_checkpoint_state_dict(state)
