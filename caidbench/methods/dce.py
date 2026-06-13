from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.covariance import OAS

from ..data.loader import build_dataloader
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, build_optimizer, freeze_module, iter_limited_train_batches


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
        for _batch_idx, batch in iter_limited_train_batches(trainer, train_loader):
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
            for key, value in method.train_metrics(out).items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n += 1
        if scheduler is not None:
            scheduler.step()
        if totals:
            metrics = {key: value / max(n, 1) for key, value in totals.items()}
            trainer.log_train_metrics(metrics, task=task, epoch=epoch + 1, epochs=epochs, optimizer=optimizer)


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


class CosineLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sigma: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.sigma = nn.Parameter(torch.ones(1)) if sigma else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
        return logits * self.sigma if self.sigma is not None else logits


class DCEExpert(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, max(dim // 2, 1)))
        self.head = CosineLinear(max(dim // 2, 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.float()))


class DCESelector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = max(int(hidden_dim if hidden_dim is not None else int(in_dim) // 2), 1)
        self.net = nn.Sequential(nn.Linear(int(in_dim), int(hidden_dim)), nn.ReLU(inplace=True), nn.Linear(int(hidden_dim), int(out_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


@register_method("dce")
class DCEMethod(FrozenFeatureMethod):
    """Dual-Balance Collaborative Experts for imbalanced DIL."""

    def __init__(
        self,
        implementation: str = "official_compatible",
        total_sessions: int = 7,
        bal_epoch: int = 10,
        selector_epoch: int = 10,
        selector_lr: float = 0.01,
        num_sampled_pcls: int = 256,
        use_sm: bool = False,
        temp: float = 1.0,
        prompt_type: str = "no",
        prompt_length: int = 10,
        margin_sample_num: int = 10,
        zero_class_count: float = 0.1,
        use_oas_covariance: bool = True,
        share_covariance_within_task: bool = True,
        covariance_jitter: float = 0.0,
        feature_scaling_mode: int = 1,
        use_official_expert_optimizer: bool = True,
        use_test_transform_for_stats: bool = True,
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
        freeze_module(self.detector.head)
        self.implementation = str(implementation)
        self.total_sessions = int(total_sessions)
        self.bal_epoch = int(bal_epoch)
        self.selector_epoch = int(selector_epoch)
        self.selector_lr = float(selector_lr)
        self.num_sampled_pcls = int(num_sampled_pcls)
        self.use_sm = bool(use_sm)
        self.temp = float(temp)
        self.prompt_type = str(prompt_type).lower()
        self.prompt_length = int(prompt_length)
        self.margin_sample_num = int(margin_sample_num)
        self.zero_class_count = float(zero_class_count)
        self.use_oas_covariance = bool(use_oas_covariance)
        self.share_covariance_within_task = bool(share_covariance_within_task)
        self.covariance_jitter = float(covariance_jitter)
        self.feature_scaling_mode = int(feature_scaling_mode)
        self.use_official_expert_optimizer = bool(use_official_expert_optimizer)
        self.use_test_transform_for_stats = bool(use_test_transform_for_stats)
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
        self.prompt_pool = self._build_prompt_pool()
        self.selector = DCESelector(int(self.detector.feature_dim), 3)
        self.stats: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.current_key = "task0"
        self.current_class_counts = torch.ones(self.num_classes, dtype=torch.float32)
        self._selector_trained_for_key: str | None = None

    def _init_selector(self, num_tasks: int) -> None:
        self.selector = DCESelector(int(self.detector.feature_dim), 3 * max(int(num_tasks), 1)).to(self.device)

    def _prompt_embed_dim(self) -> int:
        model = getattr(self.detector.backbone, "model", None)
        return int(getattr(model, "embed_dim", getattr(model, "num_features", self.detector.feature_dim)))

    def _build_prompt_pool(self) -> nn.Linear | None:
        if self.prompt_type in {"no", "none", "false", "off", ""}:
            return None
        if self.prompt_type not in {"one", "all"}:
            raise ValueError("DCE prompt_type must be one of 'one', 'all', or 'no'.")
        if not self._supports_prompt_tokens():
            raise ValueError(
                "DCE prompt_type requires a timm ViT-style backbone with patch_embed/cls_token/pos_embed. "
                "Set prompt_type='no' for feature-only backbones."
            )
        return nn.Linear(self._prompt_embed_dim(), self.prompt_length, bias=False)

    def _supports_prompt_tokens(self) -> bool:
        model = getattr(self.detector.backbone, "model", None)
        required = ("patch_embed", "cls_token", "pos_embed", "pos_drop", "blocks", "norm")
        return model is not None and all(hasattr(model, name) for name in required)

    def _extract_prompted_timm_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.prompt_pool is None:
            return super().extract_features(x)
        backbone = self.detector.backbone
        model = getattr(backbone, "model", None)
        if not self._supports_prompt_tokens() or model is None:
            raise RuntimeError("DCE prompt extraction requires a timm ViT-style backbone.")

        x = model.patch_embed(x.float())
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        pos_embed = model.pos_embed.to(dtype=x.dtype, device=x.device)
        if pos_embed.shape[1] != x.shape[1]:
            raise RuntimeError(
                f"DCE prompt path expected pos_embed length {x.shape[1]}, got {pos_embed.shape[1]}. "
                "Use a fixed 224x224 ViT-B/16-style backbone for official-compatible DCE."
            )
        x = x + pos_embed
        prompt_tokens = self.prompt_pool.weight.to(dtype=x.dtype, device=x.device)
        prompt_tokens = prompt_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x[:, :1, :], prompt_tokens, x[:, 1:, :]], dim=1)

        x = model.pos_drop(x)
        patch_drop = getattr(model, "patch_drop", None)
        if patch_drop is not None:
            x = patch_drop(x)
        norm_pre = getattr(model, "norm_pre", None)
        if norm_pre is not None:
            x = norm_pre(x)
        x = model.blocks(x)
        x = model.norm(x)
        if getattr(model, "global_pool", "token") == "avg":
            x = x[:, 1:].mean(dim=1)
        else:
            x = x[:, 0]
        fc_norm = getattr(model, "fc_norm", None)
        if fc_norm is not None:
            x = fc_norm(x)
        dropout = getattr(backbone, "dropout", nn.Identity())
        proj = getattr(backbone, "proj", nn.Identity())
        return proj(dropout(x.float()))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.prompt_pool is None:
            return super().extract_features(x)
        z = self._extract_prompted_timm_features(x.to(self.device))
        return F.normalize(z, dim=-1) if self.normalize_features else z.float()

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
        if self.prompt_pool is not None:
            prompt_trainable = self.prompt_type == "all" or (self.prompt_type == "one" and int(self.current_task_id or 0) == 0)
            for p in self.prompt_pool.parameters():
                p.requires_grad_(prompt_trainable)
        for p in self.selector.parameters():
            p.requires_grad_(False)

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.current_task_id = _task_id(task)
        self.current_key = f"task{self.current_task_id}"
        self._ensure_experts(self.current_key)
        self._init_selector(len(self.naive))
        self.current_class_counts = self._counts_from_loader(train_loader)
        self._freeze_except_current()

    def _dataset_labels(self, dataset: Any) -> torch.Tensor | None:
        if dataset is None:
            return None
        for attr in ("labels", "targets"):
            labels = getattr(dataset, attr, None)
            if labels is not None:
                return torch.as_tensor(labels, dtype=torch.long)
        loaded = getattr(dataset, "loaded", None)
        indices = getattr(dataset, "indices", None)
        metadata = getattr(loaded, "metadata", None)
        if metadata is not None and indices is not None and "label" in metadata.columns:
            values = metadata.iloc[list(indices)]["label"].to_numpy()
            return torch.as_tensor(values, dtype=torch.long)
        return None

    def _counts_from_loader(self, loader: Any | None) -> torch.Tensor:
        labels = self._dataset_labels(getattr(loader, "dataset", None))
        if labels is None and loader is not None:
            ys: list[torch.Tensor] = []
            for _batch_idx, batch in iter_limited_train_batches(self, loader):
                if "y" in batch:
                    ys.append(torch.as_tensor(batch["y"]).detach().cpu())
            if ys:
                labels = torch.cat(ys, dim=0)
        if labels is None or labels.numel() == 0:
            return torch.ones(self.num_classes, dtype=torch.float32)
        return self._counts(_local_targets(labels, self.num_classes)).detach().cpu()

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        if self.use_official_expert_optimizer:
            cfg = dict(optimizer_cfg or {})
            cfg.setdefault("type", self.optimizer_type)
            cfg.setdefault("momentum", 0.9)
            if self.init_lr is not None:
                cfg["lr"] = self.init_lr
            elif self.lr is not None:
                cfg["lr"] = self.lr
            elif self.lrate is not None:
                cfg["lr"] = self.lrate
            if self.init_weight_decay is not None:
                cfg["weight_decay"] = self.init_weight_decay
            elif self.weight_decay is not None:
                cfg["weight_decay"] = self.weight_decay
        else:
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        _run_minibatch_loop(self, trainer, task, train_loader, optimizer, epochs, scheduler)
        stats_loader = self._build_stats_loader(trainer, task, train_loader)
        features, labels = self.collect_features(stats_loader)
        self._update_stats(self.current_key, features, labels)
        self._train_selector_from_stats()
        self._selector_trained_for_key = self.current_key
        self._freeze_except_current()
        return True

    def _build_stats_loader(self, trainer: Any, task: Any, fallback_loader: Any) -> Any:
        """Match official DCE: collect current train features with test transforms."""
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

    def _counts(self, y: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(y.long(), minlength=self.num_classes).float().to(y.device)
        return torch.where(counts > 0, counts, torch.full_like(counts, self.zero_class_count))

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        z = self.extract_features(batch["x"])
        y = _local_targets(batch["y"], self.num_classes)
        counts = self.current_class_counts.to(y.device)
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
        if self._selector_trained_for_key == self.current_key:
            self._freeze_except_current()
            return
        if train_loader is None:
            return
        features, labels = self.collect_features(train_loader)
        self._update_stats(self.current_key, features, labels)
        self._train_selector_from_stats()
        self._freeze_except_current()

    def _update_stats(self, key: str, features: torch.Tensor, labels: torch.Tensor) -> None:
        pending: list[tuple[int, torch.Tensor, torch.Tensor, int, bool]] = []
        for cls in range(self.num_classes):
            z = features[labels == cls].float()
            if z.numel() == 0:
                continue
            mean = z.mean(dim=0)
            has_official_cov = self.use_oas_covariance and z.shape[0] >= self.margin_sample_num
            if has_official_cov:
                cov_np = OAS().fit(z.numpy()).covariance_
                cov = torch.as_tensor(cov_np, dtype=torch.float32)
            elif z.shape[0] >= max(self.margin_sample_num, 2):
                centered = z - mean
                cov = centered.t().matmul(centered) / max(z.shape[0] - 1, 1)
            else:
                cov = torch.eye(z.shape[1])
            pending.append((cls, mean.detach(), cov.detach(), int(z.shape[0]), has_official_cov))
        if not pending:
            return
        shared_cov: torch.Tensor | None = None
        if self.share_covariance_within_task:
            valid = [item[2] for item in pending if item[4]]
            if valid:
                shared_cov = torch.stack(valid, dim=0).mean(dim=0)
            else:
                shared_cov = torch.eye(pending[0][1].numel())
        for cls, mean, cov, count, _has_cov in pending:
            final_cov = shared_cov if shared_cov is not None else cov
            if self.covariance_jitter > 0:
                final_cov = final_cov + torch.eye(final_cov.shape[0]) * self.covariance_jitter
            self.stats[(key, cls)] = (mean, final_cov.detach(), count)

    def _sample_stats(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        xs = []
        ys = []
        cur_task = int(self.current_task_id or 0)
        for (key, cls), (mean, cov, _count) in self.stats.items():
            try:
                dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
                sample = dist.sample((self.num_sampled_pcls,))
            except Exception:
                sample = mean.unsqueeze(0).repeat(self.num_sampled_pcls, 1)
            task_id = int(key.replace("task", "")) if key.startswith("task") and key[4:].isdigit() else 0
            if self.feature_scaling_mode:
                rand_scaling = 0.02 * (torch.rand(sample.shape[0]) - 0.5)
                if self.feature_scaling_mode == 1:
                    factor = 1.0 + rand_scaling * max(cur_task - task_id, 0)
                elif self.feature_scaling_mode == 2:
                    factor = 1.0 + rand_scaling * max(cur_task, 0) / 2.0
                elif self.feature_scaling_mode == 4:
                    factor = 1.0 + rand_scaling * max(cur_task, 0)
                else:
                    factor = 1.0 + rand_scaling
                sample = sample / factor.clamp_min(1e-6).unsqueeze(1)
            xs.append(sample)
            ys.append(torch.full((sample.shape[0],), int(cls), dtype=torch.long))
        if not xs:
            return None
        return torch.cat(xs, dim=0).to(self.device), torch.cat(ys, dim=0).to(self.device)

    def _selector_epochs(self) -> int:
        run_epochs = max(int(self.bal_epoch), 1)
        if run_epochs < 5 and int(self.current_task_id or 0) == 0:
            run_epochs *= 2
        return run_epochs

    def _train_selector_from_stats(self) -> None:
        keys = list(self.naive.keys())
        if not keys:
            return
        for p in self.selector.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(self.selector.parameters(), lr=self.selector_lr, momentum=0.9, weight_decay=self.weight_decay or 2e-4)
        run_epochs = self._selector_epochs()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=run_epochs)
        self.selector.train()
        for _epoch in range(run_epochs):
            payload = self._sample_stats()
            if payload is None:
                break
            x, y = payload
            order = torch.randperm(x.size(0), device=x.device)
            x = x[order]
            y = y[order]
            for start in range(0, x.size(0), self.num_sampled_pcls):
                inp = x[start : start + self.num_sampled_pcls]
                tgt = y[start : start + self.num_sampled_pcls]
                expert_logits = self._all_expert_logits(inp, keys).detach()
                weights = self.selector(inp)[:, : 3 * len(keys)]
                weights = F.softmax(weights * self.temp, dim=1) if self.use_sm else weights
                logits = torch.einsum("be,bec->bc", weights, expert_logits)
                loss = F.cross_entropy(logits, tgt)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scheduler.step()
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
        weights = F.softmax(weights * self.temp, dim=1) if self.use_sm else weights
        logits = torch.einsum("be,bec->bc", weights, expert_logits)
        return {"logits": logits, "features": z, "expert_weights": weights.detach()}
