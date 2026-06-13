from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from ..registry import register_method
from .base import ContinualMethod, batch_to_device, build_optimizer, freeze_module, iter_limited_train_batches


_CP_PROMPT_DEFAULT_CLASS_NAMES = ("real", "fake")


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


class OfficialCPTextEncoder(nn.Module):
    """Official CP-Prompt text encoder path for the OpenAI CLIP fork."""

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.to(device=prompts.device, dtype=self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


class OfficialCPPromptLearner(nn.Module):
    """CoOp-style class text prompts used by the released CP-Prompt code."""

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer: Any,
        class_names: Sequence[str],
        n_ctx: int = 16,
        ctx_init: str = "",
        class_token_position: str = "end",
        class_specific_contexts: bool = False,
    ) -> None:
        super().__init__()
        class_names = [str(name).replace("_", " ") for name in class_names]
        n_cls = len(class_names)
        dtype = clip_model.dtype
        ctx_dim = int(clip_model.ln_final.weight.shape[0])
        if ctx_init:
            prompt_prefix = str(ctx_init).replace("_", " ")
            n_ctx = len(prompt_prefix.split(" "))
            tokenized_init = tokenizer(prompt_prefix)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_init.to(clip_model.token_embedding.weight.device)).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + int(n_ctx), :]
        else:
            prompt_prefix = " ".join(["X"] * int(n_ctx))
            shape = (n_cls, int(n_ctx), ctx_dim) if class_specific_contexts else (int(n_ctx), ctx_dim)
            ctx_vectors = torch.empty(*shape, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        device = clip_model.token_embedding.weight.device
        self.ctx = nn.Parameter(ctx_vectors.to(device))
        try:
            from clip.simple_tokenizer import SimpleTokenizer as _SimpleTokenizer  # type: ignore

            simple_tokenizer = _SimpleTokenizer()
            name_lens = [len(simple_tokenizer.encode(name)) for name in class_names]
        except Exception:  # pragma: no cover - optional dependency fallback
            name_lens = [len(name.split()) for name in class_names]
        prompts = [f"{prompt_prefix} {name}." for name in class_names]
        tokenized_prompts = torch.cat([tokenizer(prompt) for prompt in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :].detach())
        self.register_buffer("token_suffix", embedding[:, 1 + int(n_ctx) :, :].detach())
        self.n_cls = n_cls
        self.n_ctx = int(n_ctx)
        self.register_buffer("tokenized_prompts", tokenized_prompts.detach().long())
        self.name_lens = name_lens
        self.class_token_position = str(class_token_position)

    def forward(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            return torch.cat([prefix, ctx, suffix], dim=1)
        if self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i, name_len in enumerate(self.name_lens):
                prompts.append(
                    torch.cat(
                        [
                            prefix[i : i + 1],
                            ctx[i : i + 1, :half_n_ctx],
                            suffix[i : i + 1, :name_len],
                            ctx[i : i + 1, half_n_ctx:],
                            suffix[i : i + 1, name_len:],
                        ],
                        dim=1,
                    )
                )
            return torch.cat(prompts, dim=0)
        if self.class_token_position == "front":
            prompts = []
            for i, name_len in enumerate(self.name_lens):
                prompts.append(
                    torch.cat(
                        [
                            prefix[i : i + 1],
                            suffix[i : i + 1, :name_len],
                            ctx[i : i + 1],
                            suffix[i : i + 1, name_len:],
                        ],
                        dim=1,
                    )
                )
            return torch.cat(prompts, dim=0)
        raise ValueError(f"Unknown class_token_position={self.class_token_position!r}.")


def _official_tensor_prompt(pool_size: int, prompt_length: int, emb_dim: int) -> nn.Parameter:
    prompt = nn.Parameter(torch.empty(int(pool_size), int(prompt_length), int(emb_dim)), requires_grad=True)
    for index in range(int(pool_size)):
        nn.init.kaiming_uniform_(prompt[index], a=np.sqrt(5))
    return prompt


class OfficialCPPrefixPrompt(nn.Module):
    """Prefix prompt pool matching the released PrefixKeqV/PrefixKneqV modules."""

    def __init__(self, emb_dim: int, total_sessions: int, prompt_length: int, layers: Sequence[int], mode: str = "keqv") -> None:
        super().__init__()
        self.task_count = 0
        self.emb_dim = int(emb_dim)
        self.total_sessions = int(total_sessions)
        self.prompt_length = int(prompt_length)
        self.layers = [int(layer) for layer in layers]
        self.mode = str(mode).lower()
        if self.mode not in {"keqv", "kneqv"}:
            raise ValueError(f"Unknown prefix_tuning={mode!r}; expected 'keqv' or 'kneqv'.")
        for layer in self.layers:
            setattr(self, f"e_p_{layer}", _official_tensor_prompt(self.total_sessions, self.prompt_length, self.emb_dim))

    def process_task_count(self) -> None:
        self.task_count += 1

    def forward(self, layer: int, batch_size: int, task_id: int | torch.Tensor | None = None) -> torch.Tensor | list[torch.Tensor] | None:
        if int(layer) not in self.layers or task_id is None:
            return None
        prompt = getattr(self, f"e_p_{int(layer)}")
        if isinstance(task_id, int):
            selected = prompt[int(task_id)].expand(int(batch_size), -1, -1)
        else:
            ids = task_id.to(device=prompt.device, dtype=torch.long)
            selected = prompt.index_select(0, ids)
        if self.mode == "keqv":
            return selected
        split = int(self.prompt_length / 2)
        return [selected[:, :split, :].reshape(int(batch_size), -1, self.emb_dim), selected[:, split:, :].reshape(int(batch_size), -1, self.emb_dim)]


class OfficialCPPromptNet(nn.Module):
    """Method-layer official CP-Prompt network built around OpenAI CLIP."""

    def __init__(
        self,
        *,
        model_name: str = "ViT-B/16",
        pretrained: str | bool | None = "openai",
        total_sessions: int = 5,
        embd_dim: int = 768,
        share_prompt_length: int = 6,
        prefix_prompt_length: int = 10,
        prefix_prompt_layers: Sequence[int] = (3, 5, 6, 7, 8),
        prefix_tuning: str = "keqv",
        is_fix_share_prompt: bool = True,
        class_names: Sequence[str] = _CP_PROMPT_DEFAULT_CLASS_NAMES,
        n_ctx: int = 16,
        ctx_init: str = "",
        class_token_position: str = "end",
        class_specific_contexts: bool = False,
        precision: str = "fp16",
    ) -> None:
        super().__init__()
        try:
            import clip  # type: ignore
            from clip import clip as clip_impl  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install CAIDBench with `pip install -e .[clip]` to use CP-Prompt implementation=official. "
                "The official CP-Prompt path uses the OpenAI CLIP package, not OpenCLIP."
            ) from e

        official_name = self._official_clip_name(model_name)
        model_path = self._resolve_clip_path(clip_impl, official_name, pretrained)
        try:
            jit_model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = jit_model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        self.clip_model = clip_impl.build_model(state_dict)
        if str(precision).lower() == "fp32":
            self.clip_model.float()
        self.model_name = official_name
        self.image_encoder = self.clip_model.visual
        if not hasattr(self.image_encoder, "transformer") or not hasattr(self.image_encoder, "conv1"):
            raise TypeError("Official CP-Prompt requires a CLIP ViT visual encoder such as ViT-B/16.")
        self.text_encoder = OfficialCPTextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.is_fix_share_prompt = bool(is_fix_share_prompt)
        self.class_num = len(class_names)
        self.total_sessions = int(total_sessions)
        self.numtask = 0
        self.share_prompt = nn.Linear(int(embd_dim), int(share_prompt_length), bias=False)
        self.prefix_prompt = OfficialCPPrefixPrompt(
            emb_dim=int(embd_dim),
            total_sessions=int(total_sessions),
            prompt_length=int(prefix_prompt_length),
            layers=prefix_prompt_layers,
            mode=prefix_tuning,
        )
        self.classifier_pool = nn.ModuleList(
            [
                OfficialCPPromptLearner(
                    self.clip_model,
                    clip.tokenize,
                    class_names,
                    n_ctx=n_ctx,
                    ctx_init=ctx_init,
                    class_token_position=class_token_position,
                    class_specific_contexts=class_specific_contexts,
                )
                for _ in range(int(total_sessions))
            ]
        )
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        self.clip_model.eval()

    @staticmethod
    def _official_clip_name(model_name: str) -> str:
        aliases = {
            "vit-b-16": "ViT-B/16",
            "vit-b/16": "ViT-B/16",
            "vit-b-32": "ViT-B/32",
            "vit-b/32": "ViT-B/32",
            "rn50": "RN50",
            "rn101": "RN101",
            "rn50x4": "RN50x4",
            "rn50x16": "RN50x16",
        }
        return aliases.get(str(model_name).lower(), str(model_name))

    @staticmethod
    def _resolve_clip_path(clip_impl: Any, model_name: str, pretrained: str | bool | None) -> str:
        local_path = Path(str(pretrained)).expanduser() if isinstance(pretrained, str) else None
        if pretrained not in {None, False, "openai"} and local_path is not None and local_path.exists():
            return str(local_path)
        if model_name not in clip_impl._MODELS:
            raise ValueError(f"Unknown OpenAI CLIP model {model_name!r}.")
        download = clip_impl._download
        try:
            params = inspect.signature(download).parameters
        except (TypeError, ValueError):
            params = {}
        if "root" in params:
            return str(download(clip_impl._MODELS[model_name], root=str(Path.home() / ".cache" / "clip")))
        return str(download(clip_impl._MODELS[model_name]))

    @property
    def feature_dim(self) -> int:
        return int(self.image_encoder.output_dim)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.clip_model.eval()
        return self

    def freeze_for_task(self, task_id: int) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        current = int(task_id)
        for name, param in self.named_parameters():
            if f"classifier_pool.{current}" in name or "share_prompt" in name or "prefix_prompt" in name:
                param.requires_grad_(True)

    def _fixed_prompt_stack(self) -> torch.Tensor:
        prompts = []
        for task_id in range(self.total_sessions):
            name = f"fix_prompt_weight_{task_id}"
            if name in self._buffers:
                prompts.append(getattr(self, name))
        if not prompts:
            return self.share_prompt.weight.detach().unsqueeze(0)
        return torch.stack(prompts, dim=0)

    def snapshot_share_prompt(self, task_id: int) -> None:
        name = f"fix_prompt_weight_{int(task_id)}"
        snapshot = self.share_prompt.weight.detach().clone()
        if name in self._buffers:
            self._buffers[name] = snapshot
        else:
            self.register_buffer(name, snapshot)

    @staticmethod
    def _attention(block: nn.Module, x: torch.Tensor, prefix_prompt: torch.Tensor | list[torch.Tensor] | None = None) -> torch.Tensor:
        attn_mask = block.attn_mask.to(dtype=x.dtype, device=x.device) if getattr(block, "attn_mask", None) is not None else None
        if prefix_prompt is None:
            return block.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        if isinstance(prefix_prompt, list):
            key = torch.cat([x[:1], prefix_prompt[0], x[1:]], dim=0)
            value = torch.cat([x[:1], prefix_prompt[1], x[1:]], dim=0)
            return block.attn(x, key, value, need_weights=False, attn_mask=attn_mask)[0]
        prompt_x = torch.cat([x[:1], prefix_prompt, x[1:]], dim=0)
        return block.attn(x, prompt_x, prompt_x, need_weights=False, attn_mask=attn_mask)[0]

    def _resblock_forward(
        self,
        block: nn.Module,
        x: torch.Tensor,
        prefix_prompt: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if prefix_prompt is not None:
            if isinstance(prefix_prompt, list):
                pk = block.ln_1(self.image_encoder.ln_pre(prefix_prompt[0].to(dtype=x.dtype).permute(1, 0, 2)))
                pv = block.ln_1(self.image_encoder.ln_pre(prefix_prompt[1].to(dtype=x.dtype).permute(1, 0, 2)))
                prompt = [pk, pv]
            else:
                prompt = block.ln_1(self.image_encoder.ln_pre(prefix_prompt.to(dtype=x.dtype).permute(1, 0, 2)))
            x = x + self._attention(block, block.ln_1(x), prompt)
        else:
            x = x + self._attention(block, block.ln_1(x))
        x = x + block.mlp(block.ln_2(x))
        return x

    def _visual_forward(
        self,
        image: torch.Tensor,
        instance_tokens: torch.Tensor | None = None,
        prefix_prompt: OfficialCPPrefixPrompt | None = None,
        task_id: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = self.image_encoder
        x = visual.conv1(image.type(self.dtype))
        batch_size = x.shape[0]
        x = x.reshape(batch_size, x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(batch_size, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] != visual.positional_embedding.shape[0]:
            raise ValueError(
                "Official CP-Prompt expects CLIP ViT positional length to match the image resolution. "
                "For ViT-B/16, use 224x224 image tensors."
            )
        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(device=x.device, dtype=x.dtype) + torch.zeros(
                batch_size, 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
        x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        if instance_tokens is not None:
            x = torch.cat([x[:, :1, :], instance_tokens, x[:, 1:, :]], dim=1)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        for layer, block in enumerate(visual.transformer.resblocks):
            prompt = prefix_prompt(layer, batch_size, task_id=task_id) if prefix_prompt is not None else None
            x = self._resblock_forward(block, x, prompt)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj
        return x

    def extract_vector(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self._visual_forward(image, instance_tokens=None, prefix_prompt=None, task_id=None)
        return F.normalize(image_features, dim=-1)

    def extract_share_prompt_vector(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self._visual_forward(image, instance_tokens=self.share_prompt.weight, prefix_prompt=None, task_id=None)
        return F.normalize(image_features, dim=-1)

    def _text_features(self, task_id: int) -> torch.Tensor:
        prompts = self.classifier_pool[int(task_id)]
        text_features = self.text_encoder(prompts(), prompts.tokenized_prompts.to(prompts.token_prefix.device))
        return F.normalize(text_features, dim=-1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = self._visual_forward(
            image,
            instance_tokens=self.share_prompt.weight,
            prefix_prompt=self.prefix_prompt,
            task_id=int(self.numtask),
        )
        image_features = F.normalize(image_features, dim=-1)
        logits = self.logit_scale.exp() * image_features @ self._text_features(int(self.numtask)).t()
        return {"logits": logits, "features": image_features}

    def interface(self, image: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
        selection = selection.to(device=image.device, dtype=torch.long)
        if self.is_fix_share_prompt:
            prompt_stack = self._fixed_prompt_stack().to(device=image.device)
            instance_tokens = prompt_stack.index_select(0, selection)
        else:
            instance_tokens = self.share_prompt.weight
        image_features = self._visual_forward(
            image,
            instance_tokens=instance_tokens,
            prefix_prompt=self.prefix_prompt,
            task_id=selection,
        )
        image_features = F.normalize(image_features, dim=-1)
        logits_by_task = []
        for task_id in range(self.total_sessions):
            logits_by_task.append(self.logit_scale.exp() * image_features @ self._text_features(task_id).t())
        logits = torch.cat(logits_by_task, dim=1)
        selected = []
        for row, task_id in enumerate(selection.detach().cpu().tolist()):
            start = self.class_num * int(task_id)
            selected.append(logits[row, start : start + self.class_num])
        return torch.stack(selected, dim=0)


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


@register_method("cp_prompt")
@register_method("cp-prompt")
@register_method("cpprompt")
class CPPromptMethod(DomainRoutedFeatureMethod):
    """CP-Prompt composition-style common and personalized prompt reproduction."""

    def __init__(
        self,
        prompt_dim: int | None = None,
        common_prompt_lr_scale: float = 1.0,
        implementation: str = "feature_space",
        is_fix_share_prompt: bool = True,
        dataset: str = "cddb",
        total_sessions: int = 5,
        task_name: Sequence[str] | None = None,
        knn_k: int = 5,
        prefix_tuning: str = "keqv",
        share_prompt_length: int = 6,
        prefix_prompt_length: int = 10,
        prefix_prompt_layers: Sequence[int] = (3, 5, 6, 7, 8),
        embd_dim: int = 768,
        query_type: str = "vit_query",
        init_epoch: int | None = None,
        init_lr: float | None = None,
        init_lr_decay: float | None = None,
        init_weight_decay: float | None = None,
        epochs: int | None = None,
        lrate: float | None = None,
        lrate_decay: float | None = None,
        weight_decay: float | None = None,
        class_names: Sequence[str] | None = None,
        n_ctx: int = 16,
        ctx_init: str = "",
        class_token_position: str = "end",
        class_specific_contexts: bool = False,
        precision: str = "fp16",
        **kwargs: Any,
    ) -> None:
        self.implementation = str(implementation).lower()
        if self.implementation in {"official", "official_compatible", "prefix_one_prompt"}:
            nn.Module.__init__(self)
            self.implementation = "official"
            self.num_classes = int(kwargs.get("num_classes", len(class_names or _CP_PROMPT_DEFAULT_CLASS_NAMES)))
            self.current_task_id: int | None = None
            self.extra_cfg = dict(kwargs)
            detector_cfg = dict(kwargs.get("detector_cfg") or {})
            backbone_cfg = dict(detector_cfg.get("backbone") or {})
            model_name = str(backbone_cfg.get("model_name", backbone_cfg.get("name", "ViT-B/16")))
            pretrained = backbone_cfg.get("pretrained_path", backbone_cfg.get("checkpoint_path", backbone_cfg.get("pretrained", "openai")))
            if class_names is None:
                class_names = _CP_PROMPT_DEFAULT_CLASS_NAMES if str(dataset).lower() == "cddb" else [f"class {i}" for i in range(self.num_classes)]
            self.official_network = OfficialCPPromptNet(
                model_name=model_name,
                pretrained=pretrained,
                total_sessions=int(total_sessions),
                embd_dim=int(embd_dim),
                share_prompt_length=int(share_prompt_length),
                prefix_prompt_length=int(prefix_prompt_length),
                prefix_prompt_layers=prefix_prompt_layers,
                prefix_tuning=prefix_tuning,
                is_fix_share_prompt=is_fix_share_prompt,
                class_names=class_names,
                n_ctx=int(n_ctx),
                ctx_init=str(ctx_init),
                class_token_position=str(class_token_position),
                class_specific_contexts=bool(class_specific_contexts),
                precision=str(precision),
            )
            self.dataset = str(dataset)
            self.task_name = [str(name) for name in task_name] if task_name is not None else []
            self.total_sessions = int(total_sessions)
            self.knn_k = int(knn_k)
            self.query_type = str(query_type)
            self.init_epoch = None if init_epoch is None else int(init_epoch)
            self.init_lr = 0.01 if init_lr is None else float(init_lr)
            self.init_lr_decay = init_lr_decay
            self.init_weight_decay = 0.0005 if init_weight_decay is None else float(init_weight_decay)
            self.epochs = None if epochs is None else int(epochs)
            self.lrate = 0.01 if lrate is None else float(lrate)
            self.lrate_decay = lrate_decay
            self.weight_decay = 2e-4 if weight_decay is None else float(weight_decay)
            self.current_key = "task0"
            self._known_classes = 0
            self._freeze_official_except_current()
            return

        super().__init__(
            init_epoch=init_epoch,
            init_lr=init_lr,
            init_lr_decay=init_lr_decay,
            init_weight_decay=init_weight_decay,
            epochs=epochs,
            lrate=lrate,
            lrate_decay=lrate_decay,
            weight_decay=weight_decay,
            **kwargs,
        )
        if self.implementation not in {"feature_space", "compact"}:
            raise NotImplementedError("CP-Prompt implementation must be one of 'feature_space', 'compact', or 'official'.")
        dim = int(prompt_dim or self.detector.feature_dim)
        if dim != int(self.detector.feature_dim):
            raise ValueError("Feature-space CP-Prompt requires prompt_dim == detector.feature_dim.")
        self.common_prompt = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.common_prompt, std=0.02)
        self.personal_prompts = nn.ParameterDict()
        self.common_prompt_lr_scale = float(common_prompt_lr_scale)
        self.is_fix_share_prompt = bool(is_fix_share_prompt)

    def train(self, mode: bool = True):  # type: ignore[override]
        if getattr(self, "implementation", "") == "official":
            nn.Module.train(self, mode)
            self.official_network.train(mode)
            self.official_network.clip_model.eval()
            return self
        return super().train(mode)

    def _freeze_official_except_current(self) -> None:
        self.official_network.freeze_for_task(int(self.current_task_id or 0))

    @staticmethod
    def _official_targets(y: torch.Tensor, known_classes: int, class_num: int) -> torch.Tensor:
        y = y.long()
        if y.numel() and int(y.min()) >= int(known_classes) and int(y.max()) < int(known_classes) + int(class_num):
            return y - int(known_classes)
        return torch.remainder(y, int(class_num))

    @staticmethod
    def _official_center_name(task_id: int) -> str:
        return f"cp_prompt_official_centers_task{int(task_id)}"

    def _official_available_center_ids(self) -> list[int]:
        return [task_id for task_id in range(int(self.total_sessions)) if self._official_center_name(task_id) in self._buffers]

    def _checkpoint_excluded_prefixes(self) -> tuple[str, ...]:
        if getattr(self, "implementation", "") != "official":
            return ()
        return (
            "official_network.clip_model.",
            "official_network.image_encoder.",
            "official_network.text_encoder.",
            "official_network.logit_scale",
        )

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        state = super().state_dict()
        prefixes = self._checkpoint_excluded_prefixes()
        if not prefixes:
            return state
        return {key: value for key, value in state.items() if not key.startswith(prefixes)}

    def _prepare_checkpoint_state_modules(self, state: dict[str, torch.Tensor]) -> None:
        if getattr(self, "implementation", "") != "official":
            return
        for key, value in state.items():
            if key.startswith("cp_prompt_official_centers_task") and key not in self._buffers:
                self.register_buffer(key, torch.empty_like(value))

    def load_checkpoint_state_dict(self, state: dict[str, torch.Tensor]):
        self._prepare_checkpoint_state_modules(state)
        result = super().load_state_dict(state, strict=False)
        return self._filter_load_result(result, missing_prefixes=self._checkpoint_excluded_prefixes())

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        if getattr(self, "implementation", "") != "official":
            return super().before_task(task, train_loader)
        self.current_task_id = int(getattr(task, "task_id", task if isinstance(task, int) else 0))
        self.current_key = f"task{self.current_task_id}"
        self._known_classes = int(self.current_task_id) * int(self.official_network.class_num)
        self.official_network.numtask = int(self.current_task_id)
        self._freeze_official_except_current()

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        if getattr(self, "implementation", "") == "official":
            cfg = dict(optimizer_cfg or {})
            task_id = int(self.current_task_id or 0)
            lr = self.init_lr if task_id == 0 else self.lrate
            wd = self.init_weight_decay if task_id == 0 else self.weight_decay
            cfg["lr"] = float(lr)
            cfg["weight_decay"] = float(wd)
            cfg["type"] = "sgd"
            cfg.setdefault("momentum", 0.9)
            return build_optimizer(self.official_network.parameters(), cfg)
        return self._configure_feature_optimizer(optimizer_cfg)

    def _configure_feature_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        cfg = _official_optimizer_cfg(
            optimizer_cfg,
            0 if self.current_task_id is None else int(self.current_task_id),
            init_lr=self.init_lr,
            lr=self.lr,
            lrate=self.lrate,
            init_weight_decay=self.init_weight_decay,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
        )
        cfg.setdefault("lr", 1e-3)
        lr = float(cfg.get("lr", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        common = [self.common_prompt] if self.common_prompt.requires_grad else []
        common_ids = {id(p) for p in common}
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in common_ids]
        groups = []
        if other:
            groups.append({"params": other, "lr": lr, "weight_decay": weight_decay})
        if common:
            groups.append({"params": common, "lr": lr * self.common_prompt_lr_scale, "weight_decay": weight_decay})
        if not groups:
            raise RuntimeError("No trainable parameters for CP-Prompt optimizer")
        name = str(cfg.get("type", "sgd")).lower()
        if name == "adam":
            return torch.optim.Adam(groups)
        if name == "adamw":
            return torch.optim.AdamW(groups)
        return torch.optim.SGD(groups, momentum=float(cfg.get("momentum", 0.9)))

    def _official_task_epochs(self, trainer: Any) -> int:
        if int(self.current_task_id or 0) == 0 and self.init_epoch is not None:
            return max(int(self.init_epoch), 1)
        if self.epochs is not None:
            return max(int(self.epochs), 1)
        return max(int(getattr(trainer, "max_epochs", 1)), 1)

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        if getattr(self, "implementation", "") != "official":
            return super().fit_task(trainer, task, train_loader, val_loader)
        del val_loader
        self.train()
        optimizer = self.configure_optimizer(getattr(trainer, "optimizer_cfg", None))
        epochs = self._official_task_epochs(trainer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
        for epoch in range(epochs):
            totals: dict[str, float] = {}
            correct = 0
            total = 0
            n = 0
            lr = float(optimizer.param_groups[0]["lr"])
            for _batch_idx, batch in iter_limited_train_batches(trainer, train_loader):
                out = self.observe(batch, task)
                loss = out["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                logits = out["logits"]
                targets = out["targets"]
                correct += int(logits.argmax(dim=1).eq(targets).detach().cpu().sum())
                total += int(targets.numel())
                totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu())
                n += 1
            scheduler.step()
            metrics = {
                "loss": totals.get("loss", 0.0) / max(n, 1),
                "acc": correct / max(total, 1),
            }
            trainer.log_train_metrics(
                {
                    "cp_prompt_official_loss": metrics["loss"],
                    "cp_prompt_official_acc": metrics["acc"],
                },
                task=task,
                epoch=epoch + 1,
                epochs=epochs,
                lr=lr,
            )
        return True

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

    def _official_query_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.query_type == "share_p_query":
            return self.official_network.extract_share_prompt_vector(x)
        if self.query_type == "vit_query":
            return self.official_network.extract_vector(x)
        raise ValueError(f"Unknown CP-Prompt query_type={self.query_type!r}.")

    @torch.no_grad()
    def _store_official_centers(self, task_id: int, train_loader: Any) -> None:
        was_training = self.training
        self.eval()
        features = []
        for _batch_idx, batch in iter_limited_train_batches(self, train_loader):
            batch = batch_to_device(batch, self.device)
            x = batch["x"]
            features.append(self._official_query_features(x).detach().cpu())
        if was_training:
            self.train()
        if not features:
            centers = torch.empty(0, int(self.official_network.feature_dim))
        else:
            arr = torch.cat(features, dim=0).float().numpy()
            n_clusters = min(max(int(self.knn_k), 1), len(np.unique(arr, axis=0)))
            if n_clusters <= 1:
                centers_np = arr[:1]
            else:
                try:
                    centers_np = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(arr).cluster_centers_
                except TypeError:
                    centers_np = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(arr).cluster_centers_
            centers = torch.as_tensor(centers_np, dtype=torch.float32)
        name = self._official_center_name(task_id)
        if name in self._buffers:
            self._buffers[name] = centers.to(self.device)
        else:
            self.register_buffer(name, centers.to(self.device))

    def _route_official(self, features: torch.Tensor) -> torch.Tensor:
        center_ids = self._official_available_center_ids()
        if not center_ids:
            return torch.full((features.shape[0],), int(self.current_task_id or 0), dtype=torch.long, device=features.device)
        distance_columns = []
        routed_ids = []
        for task_id in center_ids:
            centers = getattr(self, self._official_center_name(task_id)).to(features.device, dtype=features.dtype)
            if centers.numel() == 0:
                continue
            distances = (features[:, None, :] - centers[None, :, :]).abs().sum(dim=-1).min(dim=1).values
            distance_columns.append(distances)
            routed_ids.append(task_id)
        if not distance_columns:
            return torch.full((features.shape[0],), int(self.current_task_id or 0), dtype=torch.long, device=features.device)
        chosen = torch.stack(distance_columns, dim=1).argmin(dim=1)
        return torch.tensor([routed_ids[int(i)] for i in chosen.detach().cpu()], dtype=torch.long, device=features.device)

    def _common_prompt_for(self, key: str, z: torch.Tensor) -> torch.Tensor:
        name = f"common_prompt_{key}"
        if self.is_fix_share_prompt and name in self._buffers:
            return getattr(self, name).to(device=z.device, dtype=z.dtype)
        return self.common_prompt.to(z.dtype)

    def _task_logits(self, z: torch.Tensor, key: str) -> torch.Tensor:
        composed = z + self._common_prompt_for(key, z) + self.personal_prompts[key].to(z.dtype)
        return self.classifiers[key](self.adapters[key](composed))

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if getattr(self, "implementation", "") != "official":
            return super().predict(batch)
        x = batch["x"].to(self.device)
        features = self._official_query_features(x)
        selection = self._route_official(features)
        logits = self.official_network.interface(x, selection)
        return {"logits": logits, "features": features, "task_selection": selection.detach()}

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        if getattr(self, "implementation", "") != "official":
            return super().observe(batch, task)
        batch = batch_to_device(batch, self.device)
        out = self.official_network(batch["x"])
        targets = self._official_targets(batch["y"], self._known_classes, self.official_network.class_num)
        loss = F.cross_entropy(out["logits"], targets)
        return {"loss": loss, "ce": loss.detach(), "logits": out["logits"].detach(), "targets": targets.detach()}

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if getattr(self, "implementation", "") == "official":
            task_id = int(self.current_task_id or 0)
            self.official_network.prefix_prompt.process_task_count()
            self.official_network.snapshot_share_prompt(task_id)
            self.official_network.numtask = min(task_id + 1, int(self.total_sessions) - 1)
            if train_loader is not None:
                self._store_official_centers(task_id, train_loader)
            self._freeze_official_except_current()
            return None
        if self.is_fix_share_prompt:
            name = f"common_prompt_{self.current_key}"
            snapshot = self.common_prompt.detach().clone()
            if name in self._buffers:
                self._buffers[name] = snapshot
            else:
                self.register_buffer(name, snapshot)
        super().after_task(task, train_loader)
