from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from ..registry import register_method
from .base import ContinualMethod, batch_to_device, build_optimizer, effective_train_batches, iter_limited_train_batches
from ..utils.logging import format_log_value, get_logger


def _insert_prompt_tokens(x: torch.Tensor, prompt_tokens: torch.Tensor | None, num_prefix_tokens: int = 1) -> torch.Tensor:
    if prompt_tokens is None:
        return x
    if prompt_tokens.ndim == 2:
        prompt_tokens = prompt_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
    elif prompt_tokens.ndim != 3:
        raise ValueError(f"S-Prompts visual prompts must have shape [L,D] or [B,L,D], got {tuple(prompt_tokens.shape)}")
    prompt_tokens = prompt_tokens.to(device=x.device, dtype=x.dtype)
    return torch.cat([x[:, :num_prefix_tokens], prompt_tokens, x[:, num_prefix_tokens:]], dim=1)


class PromptedTimmViTEncoder(nn.Module):
    """ViT image encoder with official S-iPrompts-style token insertion."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Install timm, or use method.backbone.type=open_clip for S-Prompts.") from e

        cfg = dict(cfg)
        cfg.pop("type", None)
        model_name = str(cfg.pop("name", cfg.pop("model_name", "vit_base_patch16_224")))
        pretrained = bool(cfg.pop("pretrained", True))
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **cfg)
        if not all(hasattr(self.model, name) for name in ["patch_embed", "_pos_embed", "blocks", "norm", "forward_head"]):
            raise TypeError(f"{model_name} is not a timm VisionTransformer compatible with S-Prompts token insertion.")
        self.out_dim = int(getattr(self.model, "num_features", getattr(self.model, "embed_dim", 768)))
        self.prompt_dim = int(getattr(self.model, "embed_dim", self.out_dim))

    def forward(self, x: torch.Tensor, prompt_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Official S-Prompts mode expects raw image tensors [B,3,H,W].")
        vit = self.model
        z = vit.patch_embed(x.float())
        z = vit._pos_embed(z)
        patch_drop = getattr(vit, "patch_drop", None)
        if patch_drop is not None:
            z = patch_drop(z)
        z = _insert_prompt_tokens(z, prompt_tokens, int(getattr(vit, "num_prefix_tokens", 1)))
        z = vit.norm_pre(z)
        z = vit.blocks(z)
        z = vit.norm(z)
        return vit.forward_head(z, pre_logits=True)


class PromptedOpenCLIPVisionEncoder(nn.Module):
    """OpenCLIP ViT encoder with the same visual prompt placement as the official repo."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        try:
            import open_clip  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Install open_clip_torch to use S-liPrompts.") from e

        cfg = dict(cfg)
        cfg.pop("type", None)
        model_name = str(cfg.pop("name", cfg.pop("model_name", "ViT-B-16")))
        pretrained = cfg.pop("pretrained", "openai")
        pretrained_name = "openai" if pretrained is True else (None if pretrained in {False, None, "", "none", "None"} else pretrained)
        self.model_name = model_name
        self.clip_model = open_clip.create_model(model_name, pretrained=pretrained_name, **cfg)
        self.visual = self.clip_model.visual
        required = ["conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "_pool"]
        if not all(hasattr(self.visual, name) for name in required):
            raise TypeError(f"{model_name} is not an OpenCLIP ViT visual encoder compatible with S-Prompts.")
        self.out_dim = int(getattr(self.visual, "output_dim", getattr(self.clip_model, "embed_dim", 768)))
        self.prompt_dim = int(self.visual.positional_embedding.shape[-1])

    def forward(self, x: torch.Tensor, prompt_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Official S-Prompts mode expects raw image tensors [B,3,H,W].")
        visual = self.visual
        dtype = visual.conv1.weight.dtype
        z = visual.conv1(x.to(dtype=dtype))
        z = z.reshape(z.shape[0], z.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(dtype=dtype)
        if cls.ndim == 1:
            cls = cls.view(1, 1, -1).expand(z.shape[0], -1, -1)
        else:
            cls = cls.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = z + visual.positional_embedding.to(dtype=dtype)
        z = _insert_prompt_tokens(z, prompt_tokens, 1)
        z = visual.patch_dropout(z)
        z = visual.ln_pre(z)
        z = visual.transformer(z)
        pooled, _tokens = visual._pool(z)
        if visual.proj is not None:
            pooled = visual.proj(pooled) if isinstance(visual.proj, nn.Linear) else pooled @ visual.proj
        return pooled


class OpenCLIPPromptLearner(nn.Module):
    """Task-local learnable CLIP text context, matching the official S-liPrompts head."""

    def __init__(self, clip_model: nn.Module, tokenizer: Any, class_names: Sequence[str], n_ctx: int = 16, ctx_init: str = "") -> None:
        super().__init__()
        class_names = [str(name).replace("_", " ") for name in class_names]
        ctx_dim = int(clip_model.token_embedding.weight.shape[1])
        dtype = clip_model.token_embedding.weight.dtype
        if ctx_init:
            prefix = ctx_init.replace("_", " ")
            n_ctx = len(prefix.split())
            tokenized_init = tokenizer(prefix)
            with torch.no_grad():
                ctx_vectors = clip_model.token_embedding(tokenized_init.to(clip_model.token_embedding.weight.device))[0, 1 : 1 + n_ctx]
            ctx_vectors = ctx_vectors.detach().to(dtype=torch.float32)
        else:
            prefix = " ".join(["X"] * int(n_ctx))
            ctx_vectors = torch.empty(int(n_ctx), ctx_dim, dtype=torch.float32)
            nn.init.normal_(ctx_vectors, std=0.02)

        prompts = [f"{prefix} {name}." for name in class_names]
        tokenized = tokenizer(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized.to(clip_model.token_embedding.weight.device)).to(dtype=dtype)
        self.ctx = nn.Parameter(ctx_vectors)
        self.register_buffer("token_prefix", embedding[:, :1, :].detach())
        self.register_buffer("token_suffix", embedding[:, 1 + int(n_ctx) :, :].detach())
        self.register_buffer("tokenized_prompts", tokenized.detach().long())

    def forward(self) -> torch.Tensor:
        ctx = self.ctx.to(device=self.token_prefix.device, dtype=self.token_prefix.dtype)
        ctx = ctx.unsqueeze(0).expand(self.token_prefix.shape[0], -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


@register_method("sprompts")
@register_method("s_prompts")
@register_method("s-prompts")
class SPromptsMethod(ContinualMethod):
    """Official S-Prompts mechanics adapted to CAIDBench's method interface."""

    def __init__(
        self,
        prompt_length: int = 10,
        num_centers: int = 5,
        prompt_init_scale: float = 0.02,
        train_backbone: bool = False,
        normalize_features: bool = True,
        head_type: str = "linear",
        logit_scale: float = 20.0,
        prompt_lr: float | None = None,
        random_state: int = 0,
        net_type: str = "sip",
        implementation: str = "official",
        backbone: dict[str, Any] | None = None,
        detector_cfg: dict[str, Any] | None = None,
        num_classes: int = 2,
        class_names: Sequence[str] | None = None,
        n_ctx: int = 16,
        ctx_init: str = "",
        label_mode: str = "auto",
        routing_metric: str = "official_l1",
        use_official_schedule: bool = False,
        init_epoch: int | None = None,
        init_lr: float | None = None,
        init_milestones: Sequence[int] | None = None,
        init_lr_decay: float | None = None,
        init_weight_decay: float | None = None,
        epochs: int | None = None,
        lrate: float | None = None,
        milestones: Sequence[int] | None = None,
        lrate_decay: float | None = None,
        weight_decay: float | None = None,
        **kwargs: Any,
    ) -> None:
        detector_cfg = dict(detector_cfg or {})
        backbone_cfg = dict(backbone or detector_cfg.get("backbone", {}))
        if str(implementation).lower() not in {"official", "image"}:
            raise ValueError("S-Prompts only supports raw-image official mode; pre-extracted feature modes were removed.")

        nn.Module.__init__(self)
        self.num_classes = int(num_classes)
        self.current_task_id: int | None = None
        self.extra_cfg = dict(kwargs)
        self.prompt_length = int(prompt_length)
        self.num_centers = int(num_centers)
        self.prompt_init_scale = float(prompt_init_scale)
        self.train_backbone = bool(train_backbone)
        self.normalize_features = bool(normalize_features)
        self.head_type = str(head_type).lower()
        self.logit_scale = float(logit_scale)
        self.prompt_lr = prompt_lr
        self.random_state = int(random_state)
        self.net_type = str(net_type).lower()
        self.label_mode = str(label_mode).lower()
        self.routing_metric = str(routing_metric).lower()
        self.use_official_schedule = bool(use_official_schedule)
        self.init_epoch = init_epoch
        self.init_lr = init_lr
        self.init_milestones = self._parse_milestones(init_milestones)
        self.init_lr_decay = init_lr_decay
        self.init_weight_decay = init_weight_decay
        self.official_epochs = epochs
        self.lrate = lrate
        self.milestones = self._parse_milestones(milestones)
        self.lrate_decay = lrate_decay
        self.weight_decay = weight_decay
        self.current_prompt_key = "task0"
        self._known_classes = 0

        if self.net_type not in {"sip", "slip"}:
            raise ValueError("S-Prompts net_type must be 'sip' or 'slip', matching the official repo.")
        if not backbone_cfg:
            backbone_cfg = {"type": "open_clip" if self.net_type == "slip" else "timm"}
        self.image_encoder = self._build_prompted_encoder(backbone_cfg)
        self.feature_dim = int(self.image_encoder.out_dim)
        self.prompt_dim = int(getattr(self.image_encoder, "prompt_dim", self.feature_dim))
        self.prompt_pool = nn.ParameterDict()
        self.classifier_pool = nn.ModuleDict()
        self.text_prompt_pool = nn.ModuleDict()
        self.class_names = list(class_names or [f"class {i}" for i in range(self.num_classes)])
        if len(self.class_names) != self.num_classes:
            raise ValueError(f"class_names length ({len(self.class_names)}) must match num_classes ({self.num_classes}).")
        if self.net_type == "slip" and not isinstance(self.image_encoder, PromptedOpenCLIPVisionEncoder):
            raise ValueError("S-liPrompts requires method.backbone.type=open_clip.")
        self.n_ctx = int(n_ctx)
        self.ctx_init = str(ctx_init)
        self._freeze_except_current()

    def _build_prompted_encoder(self, cfg: dict[str, Any]) -> nn.Module:
        kind = str(cfg.get("type", "timm")).lower()
        if kind in {"open_clip", "clip", "clip_vision"}:
            return PromptedOpenCLIPVisionEncoder(cfg)
        if kind in {"timm", "vit"}:
            return PromptedTimmViTEncoder(cfg)
        raise KeyError(f"Unknown S-Prompts backbone type: {kind}. Use 'timm' or 'open_clip'.")

    @staticmethod
    def _center_name(key: str) -> str:
        return f"sprompt_centers_{key}"

    @staticmethod
    def _parse_milestones(milestones: Sequence[int] | None) -> list[int]:
        if milestones is None:
            return []
        return [int(milestone) for milestone in milestones]

    def _checkpoint_excluded_prefixes(self) -> tuple[str, ...]:
        return ("image_encoder.",) if not self.train_backbone else ()

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        state = super().state_dict()
        prefixes = self._checkpoint_excluded_prefixes()
        if not prefixes:
            return state
        return {key: value for key, value in state.items() if not key.startswith(prefixes)}

    def _prepare_checkpoint_state_modules(self, state: dict[str, torch.Tensor]) -> None:
        task_keys: set[str] = set()
        for key, value in state.items():
            if key.startswith("prompt_pool."):
                task_keys.add(key.split(".", 2)[1])
            elif key.startswith("classifier_pool.") or key.startswith("text_prompt_pool."):
                task_keys.add(key.split(".", 2)[1])
            elif key.startswith("sprompt_centers_") and key not in self._buffers:
                self.register_buffer(key, torch.empty_like(value))
        for key in sorted(task_keys):
            self._add_official_task(key)

    def load_checkpoint_state_dict(self, state: dict[str, torch.Tensor]):
        self._prepare_checkpoint_state_modules(state)
        result = super().load_state_dict(state, strict=False)
        return self._filter_load_result(result, missing_prefixes=self._checkpoint_excluded_prefixes())

    def _add_official_task(self, key: str) -> None:
        if key in self.prompt_pool:
            return
        prompt = torch.randn(self.prompt_length, self.prompt_dim) * self.prompt_init_scale
        self.prompt_pool[key] = nn.Parameter(prompt)
        if self.net_type == "sip":
            self.classifier_pool[key] = nn.Linear(self.feature_dim, self.num_classes)
        else:
            import open_clip  # type: ignore

            tokenizer = open_clip.get_tokenizer(self.image_encoder.model_name)
            self.text_prompt_pool[key] = OpenCLIPPromptLearner(
                self.image_encoder.clip_model,
                tokenizer,
                self.class_names,
                n_ctx=self.n_ctx,
                ctx_init=self.ctx_init,
            )

    def _freeze_except_current(self) -> None:
        for p in self.image_encoder.parameters():
            p.requires_grad_(self.train_backbone)
        for key, p in self.prompt_pool.items():
            p.requires_grad_(key == self.current_prompt_key)
        for pool in [self.classifier_pool, self.text_prompt_pool]:
            for key, module in pool.items():
                for p in module.parameters():
                    p.requires_grad_(key == self.current_prompt_key)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if not self.train_backbone:
            self.image_encoder.eval()
        return self

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.current_task_id = int(getattr(task, "task_id", task if isinstance(task, int) else 0))
        self._known_classes = int(self.current_task_id) * self.num_classes
        key = f"task{self.current_task_id}"
        self._add_official_task(key)
        self.current_prompt_key = key
        self._freeze_except_current()

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        cfg = dict(optimizer_cfg or {})
        if self.use_official_schedule:
            if self.current_task_id == 0:
                if self.init_lr is not None:
                    cfg["lr"] = float(self.init_lr)
                if self.init_weight_decay is not None:
                    cfg["weight_decay"] = float(self.init_weight_decay)
            else:
                if self.lrate is not None:
                    cfg["lr"] = float(self.lrate)
                if self.weight_decay is not None:
                    cfg["weight_decay"] = float(self.weight_decay)
        if self.prompt_lr is not None:
            cfg["lr"] = self.prompt_lr
        cfg.setdefault("lr", 1e-3)
        return build_optimizer(self.parameters(), cfg)

    def _task_epochs(self, trainer: Any) -> int:
        if not self.use_official_schedule:
            return int(trainer.max_epochs)
        if self.current_task_id == 0 and self.init_epoch is not None:
            return int(self.init_epoch)
        if self.official_epochs is not None:
            return int(self.official_epochs)
        return int(trainer.max_epochs)

    def _configure_official_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        if not self.use_official_schedule:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=int(epochs))

    def _log_stage(self, message: str, logger: Any | None = None) -> None:
        logger = logger or get_logger("caidbench")
        logger.info("[SPrompts] %s", message)

    def _selector_features(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        return F.normalize(z, dim=-1) if self.normalize_features else z

    def _available_center_keys(self) -> list[str]:
        return [key for key in self.prompt_pool.keys() if self._center_name(key) in self._buffers]

    def _route(self, z: torch.Tensor, task_keys: list[str]) -> torch.Tensor:
        center_keys = self._available_center_keys()
        if not center_keys:
            current_index = task_keys.index(self.current_prompt_key) if self.current_prompt_key in task_keys else len(task_keys) - 1
            return torch.full((z.shape[0],), current_index, dtype=torch.long, device=z.device)

        z_sel = self._selector_features(z)
        distance_columns = []
        routed_keys = []
        for key in center_keys:
            centers = getattr(self, self._center_name(key)).to(z_sel.device).float()
            if centers.numel() == 0:
                continue
            if self.routing_metric in {"official", "official_l1", "l1"}:
                distances = (z_sel[:, None, :] - centers[None, :, :]).abs().sum(dim=-1).min(dim=1).values
            else:
                distances = torch.cdist(z_sel, centers).min(dim=1).values
            distance_columns.append(distances)
            routed_keys.append(key)
        if not distance_columns:
            return torch.zeros((z.shape[0],), dtype=torch.long, device=z.device)

        routed = torch.stack(distance_columns, dim=1).argmin(dim=1)
        key_to_index = {key: idx for idx, key in enumerate(task_keys)}
        selected = [key_to_index[routed_keys[int(i)]] for i in routed.detach().cpu()]
        return torch.tensor(selected, dtype=torch.long, device=z.device)

    def _local_targets(self, y: torch.Tensor) -> torch.Tensor:
        y = y.long()
        if self.label_mode == "local":
            return y
        if self.label_mode == "modulo":
            return torch.remainder(y, self.num_classes)
        if self.label_mode == "class_incremental":
            return y - self._known_classes
        if y.numel() and y.min() >= self._known_classes and y.max() < self._known_classes + self.num_classes:
            return y - self._known_classes
        if y.numel() and y.max() >= self.num_classes:
            return torch.remainder(y, self.num_classes)
        return y

    def _encode_text_prompts(self, learner: OpenCLIPPromptLearner) -> torch.Tensor:
        from open_clip.transformer import text_global_pool  # type: ignore

        clip_model = self.image_encoder.clip_model
        prompts = learner()
        tokenized = learner.tokenized_prompts.to(prompts.device)
        cast_dtype = clip_model.transformer.get_cast_dtype()
        z = prompts.to(cast_dtype) + clip_model.positional_embedding.to(device=prompts.device, dtype=cast_dtype)
        z = clip_model.transformer(z, attn_mask=clip_model.attn_mask)
        z = clip_model.ln_final(z)
        z = text_global_pool(z, tokenized, clip_model.text_pool_type, getattr(clip_model, "text_eos_id", None))
        if clip_model.text_projection is not None:
            z = clip_model.text_projection(z) if isinstance(clip_model.text_projection, nn.Linear) else z @ clip_model.text_projection
        return z

    def _head_logits(self, z: torch.Tensor, key: str) -> torch.Tensor:
        if self.net_type == "sip":
            return self.classifier_pool[key](z)
        z = F.normalize(z, dim=-1)
        text_features = F.normalize(self._encode_text_prompts(self.text_prompt_pool[key]).to(dtype=z.dtype), dim=-1)
        scale = self.image_encoder.clip_model.logit_scale.exp().to(dtype=z.dtype)
        return scale * z @ text_features.t()

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        task_keys = list(self.prompt_pool.keys())
        if not task_keys:
            raise RuntimeError("S-Prompts task bank is empty; call before_task first.")
        selector_z = self._selector_features(self.image_encoder(x, None))
        selection = self._route(selector_z, task_keys)
        prompt_batch = torch.stack([self.prompt_pool[task_keys[int(i)]] for i in selection.detach().cpu()], dim=0).to(x.device)
        z = self.image_encoder(x, prompt_batch)
        logits_by_task = torch.stack([self._head_logits(z, key) for key in task_keys], dim=1)
        logits = logits_by_task[torch.arange(z.shape[0], device=z.device), selection]
        return {"logits": logits, "features": selector_z, "task_selection": selection.detach()}

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        z = self.image_encoder(batch["x"], self.prompt_pool[self.current_prompt_key])
        logits = self._head_logits(z, self.current_prompt_key)
        loss = F.cross_entropy(logits, self._local_targets(batch["y"]))
        return {"loss": loss, "ce": loss.detach()}

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        if not self.use_official_schedule:
            return False
        self.train()
        optimizer = self.configure_optimizer(getattr(trainer, "optimizer_cfg", None))
        epochs = self._task_epochs(trainer)
        scheduler = self._configure_official_scheduler(optimizer, epochs)
        num_batches = effective_train_batches(trainer, train_loader)
        train_log_interval = int(getattr(trainer, "train_log_interval", 0) or 0)
        task_label = getattr(task, "name", task)
        formatted_task_label = format_log_value(task_label)
        self._log_stage(
            f"fit_task start task={formatted_task_label} task_index={getattr(task, 'task_id', self.current_task_id)} epochs={epochs}",
            getattr(trainer, "logger", None),
        )
        for epoch in range(epochs):
            self._log_stage(
                f"fit_task epoch_begin epoch={epoch + 1}/{epochs} task={formatted_task_label}",
                getattr(trainer, "logger", None),
            )
            epoch_started_at = time.monotonic()
            totals: dict[str, float] = {}
            n = 0
            for batch_idx, batch in iter_limited_train_batches(trainer, train_loader):
                out = self.observe(batch, task)
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                for key, value in self.train_metrics(out).items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                n += 1
                if train_log_interval > 0 and batch_idx % train_log_interval == 0:
                    metrics = {k: v / max(n, 1) for k, v in totals.items()}
                    trainer.log_train_metrics(
                        metrics,
                        task=task,
                        epoch=epoch + 1,
                        epochs=epochs,
                        optimizer=optimizer,
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        started_at=epoch_started_at,
                    )
            if totals:
                metrics = {k: v / max(n, 1) for k, v in totals.items()}
                trainer.log_train_metrics(
                    metrics,
                    task=task,
                    epoch=epoch + 1,
                    epochs=epochs,
                    optimizer=optimizer,
                    batch_idx=n,
                    num_batches=num_batches,
                    started_at=epoch_started_at,
                )
                self._log_stage(
                    f"fit_task epoch_end epoch={epoch + 1}/{epochs} loss={metrics.get('loss', 0.0):.4f} "
                    f"lr={float(optimizer.param_groups[0]['lr']):.6g}",
                    getattr(trainer, "logger", None),
                )
            if scheduler is not None:
                scheduler.step()
        self._log_stage(f"fit_task done task={formatted_task_label}", getattr(trainer, "logger", None))
        return True

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        features = self._selector_features(features).detach().cpu()
        if features.numel() == 0:
            return features.reshape(0, self.feature_dim)
        arr = features.numpy()
        unique = np.unique(arr, axis=0)
        n_clusters = min(max(self.num_centers, 1), len(unique))
        if n_clusters == 1:
            centers = unique[:1]
        else:
            try:
                km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto").fit(unique)
            except TypeError:  # scikit-learn < 1.4
                km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10).fit(unique)
            centers = km.cluster_centers_
        return self._selector_features(torch.as_tensor(centers, dtype=features.dtype))

    def _store_centers(self, key: str, centers: torch.Tensor) -> None:
        name = self._center_name(key)
        centers = centers.detach().to(self.device)
        if name in self._buffers:
            setattr(self, name, centers)
        else:
            self.register_buffer(name, centers)

    @torch.no_grad()
    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is None:
            return None
        was_training = self.training
        self.eval()
        total_batches = effective_train_batches(self, train_loader)
        task_label = getattr(task, "name", task)
        self._log_stage(
            f"after_task start task={format_log_value(task_label)} task_index={self.current_task_id} stage=extract_features"
        )
        features = []
        for batch_idx, batch in iter_limited_train_batches(self, train_loader):
            x = batch["x"].to(self.device)
            features.append(self.image_encoder(x, None).detach().cpu())
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % 50 == 0:
                self._log_stage(f"after_task extracting_features progress={batch_idx}/{total_batches} task_index={self.current_task_id}")
        self._log_stage(f"after_task feature_collection_done task_index={self.current_task_id} stage=cluster")
        if features:
            all_features = torch.cat(features, dim=0)
            self._log_stage(
                f"after_task cluster_start task_index={self.current_task_id} n_clusters={self.num_centers} "
                f"all_samples={all_features.shape[0]}",
            )
            self._store_centers(self.current_prompt_key, self._cluster_features(all_features))
        else:
            self._log_stage(f"after_task cluster_start task_index={self.current_task_id} n_clusters={self.num_centers} all_samples=0")
        if was_training:
            self.train()
        self._freeze_except_current()
        self._log_stage(f"after_task done task_index={self.current_task_id}")
        return None
