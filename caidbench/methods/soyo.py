from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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


class SOYOSelector(nn.Module):
    def __init__(self, in_dim: int, max_tasks: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(max_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.float())


def _insert_soyo_prompt_tokens(x: torch.Tensor, prompt_tokens: torch.Tensor | None, num_prefix_tokens: int = 1) -> torch.Tensor:
    if prompt_tokens is None:
        return x
    if prompt_tokens.ndim == 2:
        prompt_tokens = prompt_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
    elif prompt_tokens.ndim != 3:
        raise ValueError(f"SOYO prompt tokens must have shape [L,D] or [B,L,D], got {tuple(prompt_tokens.shape)}")
    prompt_tokens = prompt_tokens.to(device=x.device, dtype=x.dtype)
    return torch.cat([x[:, :num_prefix_tokens], prompt_tokens, x[:, num_prefix_tokens:]], dim=1)


class SOYOPromptedTimmViTEncoder(nn.Module):
    """Official SOYO ViT encoder: prompt tokens plus per-block down/up offsets."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Install timm to use SOYO implementation=official with net_type=soyo_vit.") from e

        cfg = dict(cfg)
        cfg.pop("type", None)
        model_name = str(cfg.pop("name", cfg.pop("model_name", "vit_base_patch16_224")))
        pretrained = bool(cfg.pop("pretrained", True))
        # CAID wrapper backbones use out_dim for projection; raw timm ViTs do not accept it.
        cfg.pop("out_dim", None)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **cfg)
        if not all(hasattr(self.model, name) for name in ["patch_embed", "cls_token", "pos_embed", "blocks", "norm"]):
            raise TypeError(f"{model_name} is not a timm VisionTransformer compatible with official SOYO.")
        self.out_dim = int(getattr(self.model, "num_features", getattr(self.model, "embed_dim", 768)))
        self.prompt_dim = int(getattr(self.model, "embed_dim", self.out_dim))
        self.num_layers = len(self.model.blocks)

    def _pos_embed(self, z: torch.Tensor) -> torch.Tensor:
        vit = self.model
        cls = vit.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat((cls, z), dim=1)
        if z.shape[1] != vit.pos_embed.shape[1]:
            raise ValueError(
                f"SOYO token count ({z.shape[1]}) does not match positional embedding count ({vit.pos_embed.shape[1]}). "
                "Set the timm backbone img_size to match the input transform."
            )
        return z + vit.pos_embed.to(device=z.device, dtype=z.dtype)

    def forward(
        self,
        x: torch.Tensor,
        img_prompt: torch.Tensor | None = None,
        up_pool: nn.ModuleList | None = None,
        down_pool: nn.ModuleList | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Official SOYO expects raw image tensors [B,3,H,W].")
        vit = self.model
        z = vit.patch_embed(x.float())
        z = self._pos_embed(z)
        z = _insert_soyo_prompt_tokens(z, img_prompt, int(getattr(vit, "num_prefix_tokens", 1)))
        if hasattr(vit, "pos_drop"):
            z = vit.pos_drop(z)
        if hasattr(vit, "patch_drop"):
            z = vit.patch_drop(z)
        if hasattr(vit, "norm_pre"):
            z = vit.norm_pre(z)

        for layer, block in enumerate(vit.blocks):
            z = block(z)
            if up_pool is not None:
                if down_pool is None:
                    raise ValueError("SOYO up_pool requires a matching down_pool.")
                z = z + up_pool[layer](down_pool[layer](z))

        z = vit.norm(z)
        if hasattr(vit, "forward_head"):
            return vit.forward_head(z, pre_logits=True)
        return z[:, 0]


class OfficialSOYOViT(nn.Module):
    """Released SOYO-ViT network structure adapted to CAIDBench batches."""

    def __init__(
        self,
        *,
        backbone_cfg: dict[str, Any],
        total_sessions: int,
        num_classes: int,
        prompt_length: int = 10,
        hidden_dim: int = 16,
        image_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = SOYOPromptedTimmViTEncoder(backbone_cfg)
        self.total_sessions = int(total_sessions)
        self.class_num = int(num_classes)
        self.feature_dim = int(self.image_encoder.out_dim)
        prompt_dim = int(self.image_encoder.prompt_dim)
        if image_dim is not None and int(image_dim) != prompt_dim:
            raise ValueError(f"SOYO image_dim={image_dim} does not match backbone prompt dim={prompt_dim}.")

        self.classifier = nn.ModuleList([nn.Linear(self.feature_dim, self.class_num, bias=True) for _ in range(self.total_sessions)])
        self.prompt_pool = nn.ModuleList([nn.Linear(prompt_dim, int(prompt_length), bias=False) for _ in range(self.total_sessions)])
        self.down_pool = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(prompt_dim, int(hidden_dim)) for _ in range(self.image_encoder.num_layers)])
                for _ in range(self.total_sessions)
            ]
        )
        self.up_pool = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(int(hidden_dim), prompt_dim) for _ in range(self.image_encoder.num_layers)])
                for _ in range(self.total_sessions)
            ]
        )
        for task_down, task_up in zip(self.down_pool, self.up_pool):
            for down, up in zip(task_down, task_up):
                nn.init.xavier_uniform_(down.weight)
                nn.init.zeros_(down.bias)
                nn.init.xavier_uniform_(up.weight)
                nn.init.zeros_(up.bias)

    def extract_vector(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        return F.normalize(image_features, dim=-1)

    def forward_task(self, image: torch.Tensor, task_index: int) -> dict[str, torch.Tensor]:
        task_index = int(task_index)
        img_prompt = self.prompt_pool[task_index].weight
        image_features = self.image_encoder(
            image,
            img_prompt=img_prompt,
            up_pool=self.up_pool[task_index],
            down_pool=self.down_pool[task_index],
        )
        return {"logits": self.classifier[task_index](image_features), "features": image_features}

    def interface(self, image: torch.Tensor, selection: torch.Tensor) -> dict[str, torch.Tensor]:
        selection = selection.long()
        prompt_stack = torch.stack([prompt.weight for prompt in self.prompt_pool], dim=0).to(image.device)
        instance_batch = prompt_stack[selection]

        feature_list = []
        for task_index in range(self.total_sessions):
            feature_list.append(
                self.image_encoder(
                    image,
                    img_prompt=instance_batch,
                    up_pool=self.up_pool[task_index],
                    down_pool=self.down_pool[task_index],
                )
            )
        stacked_features = torch.stack(feature_list, dim=0)
        batch_index = torch.arange(image.shape[0], device=image.device)
        image_features = stacked_features[selection, batch_index]

        logits_list = [classifier(image_features) for classifier in self.classifier]
        stacked_logits = torch.stack(logits_list, dim=0)
        return {"logits": stacked_logits[selection, batch_index], "features": image_features}


@register_method("soyo")
class SOYOMethod(DomainRoutedFeatureMethod):
    """SOYO parameter-isolation plus learned domain selector."""

    method_name = "soyo"

    def __init__(
        self,
        total_sessions: int = 5,
        gmm_components: int = 2,
        soyo_epoch: int = 30,
        soyo_lr: float = 0.1,
        soyo_weight_decay: float = 2e-4,
        resample_per_domain: int = 256,
        selector_batch_size: int = 128,
        normalize_selector_features: bool = True,
        implementation: str = "feature_space",
        net_type: str = "soyo_vit",
        backbone: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        implementation_name = str(implementation).lower()
        net_type_name = str(net_type).lower()
        if implementation_name in {"official", "image"}:
            nn.Module.__init__(self)
            self.implementation = "official"
            self.net_type = net_type_name
            self.num_classes = int(kwargs.pop("num_classes", 2))
            self.current_task_id: int | None = None
            detector_cfg = dict(kwargs.pop("detector_cfg", {}) or {})
            backbone_cfg = dict(backbone or detector_cfg.get("backbone", {}) or {})
            if not backbone_cfg:
                backbone_cfg = {"type": "timm", "name": "vit_base_patch16_224", "pretrained": True}
            if self.net_type != "soyo_vit":
                raise NotImplementedError("SOYO implementation=official currently supports net_type=soyo_vit.")
            image_dim_raw = kwargs.get("image_dim")
            image_dim = None if image_dim_raw is None else int(image_dim_raw)
            self.official_network = OfficialSOYOViT(
                backbone_cfg=backbone_cfg,
                total_sessions=total_sessions,
                num_classes=self.num_classes,
                prompt_length=int(kwargs.get("prompt_length", 10)),
                hidden_dim=int(kwargs.get("hidden_dim", 16)),
                image_dim=image_dim,
            )
            self.feature_dim = int(self.official_network.feature_dim)
            soyo_dim = int(kwargs.get("soyo_dim", self.feature_dim))
            if soyo_dim != self.feature_dim:
                raise ValueError(f"SOYO soyo_dim={soyo_dim} does not match backbone feature dim={self.feature_dim}.")
            self.total_sessions = int(total_sessions)
            self.gmm_components = int(gmm_components)
            self.soyo_epoch = int(soyo_epoch)
            self.soyo_lr = float(soyo_lr)
            self.soyo_weight_decay = float(soyo_weight_decay)
            self.resample_per_domain = int(resample_per_domain)
            self.selector_batch_size = int(selector_batch_size)
            self.normalize_selector_features = bool(normalize_selector_features)
            self.selector = SOYOSelector(self.feature_dim, self.total_sessions)
            self.domain_compression: list[GaussianMixture] = []
            self._known_classes = 0
            self.extra_cfg = dict(kwargs)
            self._freeze_official_except_current()
            return

        super().__init__(**kwargs)
        self.implementation = "feature_space"
        self.net_type = net_type_name
        self.total_sessions = int(total_sessions)
        self.gmm_components = int(gmm_components)
        self.soyo_epoch = int(soyo_epoch)
        self.soyo_lr = float(soyo_lr)
        self.soyo_weight_decay = float(soyo_weight_decay)
        self.resample_per_domain = int(resample_per_domain)
        self.selector_batch_size = int(selector_batch_size)
        self.normalize_selector_features = bool(normalize_selector_features)
        self.selector = SOYOSelector(int(self.detector.feature_dim), self.total_sessions)
        self.gmms: dict[str, GaussianMixture] = {}

    def train(self, mode: bool = True):  # type: ignore[override]
        if self.implementation != "official":
            return super().train(mode)
        nn.Module.train(self, mode)
        self.official_network.eval()
        return self

    def _selector_features(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        return F.normalize(z, dim=-1) if self.normalize_selector_features else z

    def _freeze_official_except_current(self) -> None:
        if self.current_task_id is None:
            current = -1
        else:
            current = int(self.current_task_id)
        for p in self.official_network.parameters():
            p.requires_grad_(False)
        if 0 <= current < self.total_sessions:
            for module in [
                self.official_network.prompt_pool[current],
                self.official_network.classifier[current],
                self.official_network.down_pool[current],
                self.official_network.up_pool[current],
            ]:
                for p in module.parameters():
                    p.requires_grad_(True)
        for p in self.selector.parameters():
            p.requires_grad_(False)

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        if self.implementation != "official":
            return super().before_task(task, train_loader)
        self.current_task_id = _task_id(task)
        if self.current_task_id >= self.total_sessions:
            raise ValueError(f"SOYO task_id={self.current_task_id} exceeds total_sessions={self.total_sessions}.")
        self._known_classes = int(self.current_task_id) * self.num_classes
        self._freeze_official_except_current()
        return None

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        if self.implementation != "official":
            return super().configure_optimizer(optimizer_cfg)
        task_id = 0 if self.current_task_id is None else int(self.current_task_id)
        cfg = _official_optimizer_cfg(
            optimizer_cfg,
            task_id,
            init_lr=self.extra_cfg.get("init_lr"),
            lr=self.extra_cfg.get("lr"),
            lrate=self.extra_cfg.get("lrate"),
            init_weight_decay=self.extra_cfg.get("init_weight_decay"),
            weight_decay=self.extra_cfg.get("weight_decay"),
            optimizer_type=str(self.extra_cfg.get("optimizer_type", "sgd")),
        )
        cfg.setdefault("lr", 1e-3)
        return build_optimizer(self.parameters(), cfg)

    def _official_targets(self, y: torch.Tensor) -> torch.Tensor:
        return _local_targets(y, self.num_classes)

    def _official_domain_targets(self, y: torch.Tensor, count: int) -> torch.Tensor:
        y = y.long()
        if y.numel() and int(y.max()) >= self.num_classes:
            return torch.div(y, self.num_classes, rounding_mode="floor")
        task_id = 0 if self.current_task_id is None else int(self.current_task_id)
        return torch.full((int(count),), task_id, dtype=torch.long, device=y.device)

    def _official_observe(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.current_task_id is None:
            raise RuntimeError("SOYO official observe called before before_task.")
        batch = batch_to_device(batch, self.device)
        out = self.official_network.forward_task(batch["x"], int(self.current_task_id))
        loss = F.cross_entropy(out["logits"], self._official_targets(batch["y"]))
        return {"loss": loss, "ce": loss.detach(), "logits": out["logits"].detach()}

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        if self.implementation != "official":
            return super().fit_task(trainer, task, train_loader, val_loader)
        del val_loader
        epochs = _official_task_epochs(
            trainer,
            _task_id(task),
            self.extra_cfg.get("init_epoch"),
            self.extra_cfg.get("epochs"),
        )
        optimizer = self.configure_optimizer(getattr(trainer, "optimizer_cfg", None))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max(int(epochs), 1))
        self.train()
        for epoch in range(epochs):
            totals: dict[str, float] = {}
            n = 0
            for batch in train_loader:
                out = self._official_observe(batch)
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                for key, value in out.items():
                    if key == "logits":
                        continue
                    if torch.is_tensor(value) and value.ndim == 0:
                        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
                n += 1
            scheduler.step()
            if totals:
                metrics = {key: value / max(n, 1) for key, value in totals.items()}
                trainer.logger.info(
                    "task=%s epoch=%d/%d %s",
                    task.name,
                    epoch + 1,
                    epochs,
                    ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
                )
                trainer.log_metrics(
                    {
                        **{f"train/{key}": value for key, value in metrics.items()},
                        "train/task_index": float(_task_id(task)),
                        "train/epoch": epoch + 1,
                        "train/lr": float(scheduler.get_last_lr()[0]),
                    }
                )
        return True

    def _route(self, z: torch.Tensor) -> torch.Tensor:
        keys = list(self.adapters.keys())
        if not keys:
            return torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        logits = self.selector(self._selector_features(z))
        return logits[:, : len(keys)].argmax(dim=1)

    @torch.no_grad()
    def _collect_official_features(self, loader: Any) -> torch.Tensor:
        was_training = self.training
        self.eval()
        features = []
        for batch in loader:
            batch = batch_to_device(batch, self.device)
            features.append(self.official_network.extract_vector(batch["x"]).detach().cpu())
        if was_training:
            self.train()
        if not features:
            return torch.empty(0, self.feature_dim)
        return torch.cat(features, dim=0)

    def _train_official_selector(self, train_loader: Any) -> None:
        for p in self.selector.parameters():
            p.requires_grad_(True)

        old_features = None
        old_targets = None
        old_count = len(self.domain_compression)
        if old_count > 0:
            sampled_features = []
            sampled_targets = []
            n_samples = max(len(train_loader.dataset), 1)
            for domain_index, compression in enumerate(self.domain_compression):
                samples, _ = compression.sample(n_samples)
                sampled_features.append(samples)
                sampled_targets.append(np.full((n_samples,), domain_index))
            old_features = torch.as_tensor(np.vstack(sampled_features), dtype=torch.float32)
            old_targets = torch.as_tensor(np.hstack(sampled_targets), dtype=torch.long)

        batch_size = int(getattr(train_loader, "batch_size", None) or self.selector_batch_size or 1)
        num_new = max(batch_size // (1 + old_count), 1)
        num_old = max(batch_size - num_new, 0)
        optimizer = torch.optim.SGD(self.selector.parameters(), momentum=0.9, lr=self.soyo_lr, weight_decay=self.soyo_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max(self.soyo_epoch, 1))
        self.selector.train()

        for _epoch in range(max(self.soyo_epoch, 1)):
            for batch in train_loader:
                batch = batch_to_device(batch, self.device)
                with torch.no_grad():
                    features = self.official_network.extract_vector(batch["x"]).float()
                domain_targets = self._official_domain_targets(batch["y"], features.shape[0])
                if old_features is not None and old_targets is not None:
                    select_new = torch.randperm(features.shape[0], device=features.device)[:num_new]
                    select_old = torch.randperm(old_features.shape[0])[:num_old]
                    final_features = torch.cat([features[select_new], old_features[select_old].to(self.device)], dim=0)
                    final_targets = torch.cat([domain_targets[select_new], old_targets[select_old].to(self.device)], dim=0)
                else:
                    final_features = features
                    final_targets = domain_targets
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.selector(final_features), final_targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
        for p in self.selector.parameters():
            p.requires_grad_(False)

    def _save_official_compression(self, train_loader: Any) -> None:
        features = self._collect_official_features(train_loader)
        if features.numel() == 0:
            return
        n_components = min(max(1, self.gmm_components), int(features.shape[0]))
        compression = GaussianMixture(n_components=n_components, covariance_type="full", init_params="kmeans").fit(features.numpy())
        task_id = 0 if self.current_task_id is None else int(self.current_task_id)
        if task_id < len(self.domain_compression):
            self.domain_compression[task_id] = compression
        else:
            self.domain_compression.append(compression)

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.implementation != "official":
            return super().predict(batch)
        x = batch["x"].to(self.device)
        selector_features = self.official_network.extract_vector(x)
        selection = self.selector(selector_features).argmax(dim=1)
        out = self.official_network.interface(x, selection)
        return {"logits": out["logits"], "features": selector_features, "task_selection": selection.detach()}

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        if self.implementation != "official":
            return super().observe(batch, task)
        return self._official_observe(batch)

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if self.implementation == "official":
            if train_loader is None:
                return None
            self._train_official_selector(train_loader)
            self._save_official_compression(train_loader)
            self._freeze_official_except_current()
            return None
        if train_loader is None:
            return
        features, _labels = self.collect_features(train_loader)
        features = self._selector_features(features.to(self.device)).cpu()
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
            count = max(int(current_features.shape[0]), self.resample_per_domain, 1)
            samples, _ = gmm.sample(count)
            features.append(torch.as_tensor(samples, dtype=torch.float32, device=self.device))
            labels.append(torch.full((count,), key_to_id[key], dtype=torch.long, device=self.device))
        features.append(current_features.float())
        labels.append(torch.full((current_features.shape[0],), key_to_id[current_key], dtype=torch.long, device=self.device))
        x = torch.cat(features, dim=0)
        y = torch.cat(labels, dim=0)
        optimizer = torch.optim.SGD(self.selector.parameters(), lr=self.soyo_lr, momentum=0.9, weight_decay=self.soyo_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.soyo_epoch, 1))
        self.selector.train()
        batch_size = max(self.selector_batch_size, 1)
        for _epoch in range(max(self.soyo_epoch, 1)):
            order = torch.randperm(x.shape[0], device=x.device)
            for start in range(0, x.shape[0], batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.selector(x[idx])[:, : len(keys)], y[idx])
                loss.backward()
                optimizer.step()
            scheduler.step()
