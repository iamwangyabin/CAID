from __future__ import annotations

import inspect
import math
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from ..data.object_labels import parse_object_labels
from ..registry import register_method
from .base import ContinualMethod, build_optimizer


def _clip_model_dtype(clip_model: nn.Module) -> torch.dtype:
    dtype = getattr(clip_model, "dtype", None)
    if dtype is not None:
        return dtype
    visual = getattr(clip_model, "visual", None)
    conv1 = getattr(visual, "conv1", None)
    if conv1 is not None:
        return conv1.weight.dtype
    return clip_model.token_embedding.weight.dtype


class PromptLearner(nn.Module):
    """Official Prompt2Guard learnable text/image prompt pair."""

    def __init__(self, clip_model: nn.Module, k: int) -> None:
        super().__init__()
        self.k = int(k)
        if self.k < 1:
            raise ValueError("Prompt2Guard requires K >= 1.")
        dtype = _clip_model_dtype(clip_model)
        device = clip_model.token_embedding.weight.device
        text_dim = int(clip_model.ln_final.weight.shape[0])
        visual_dim = int(clip_model.visual.class_embedding.shape[-1])

        eot_id = min(49407, int(clip_model.token_embedding.num_embeddings) - 1)
        text_token = clip_model.token_embedding(torch.tensor([eot_id], device=device)).detach().repeat(self.k, 1)
        text_noise = F.normalize(torch.randn(self.k, text_dim, device=device), dim=-1)
        self.text_prompt = nn.Parameter((text_token + 0.1 * text_noise).to(dtype=dtype))

        visual_token = clip_model.visual.class_embedding.detach().view(1, -1).repeat(self.k, 1)
        visual_noise = F.normalize(torch.randn(self.k, visual_dim, device=device), dim=-1)
        self.img_prompt = nn.Parameter((visual_token + 0.1 * visual_noise).to(dtype=dtype))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.text_prompt, self.img_prompt


class SliNet(nn.Module):
    """Prompt2Guard SliNet adapted from the official repository.

    The network owns a frozen CLIP ViT and one PromptLearner per incremental
    task. Forward training uses only the current prompt learner; inference
    concatenates all seen prompt learners and applies prototype-weighted task
    aggregation.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str | bool | None = "openai",
        k: int = 7,
        topk_classes: int = 5,
        ensembling: tuple[bool, bool, bool, bool] | list[bool] = (False, False, True, False),
        class_names: tuple[str, str] | list[str] = ("real", "fake"),
        precision: str = "fp16",
    ) -> None:
        super().__init__()
        try:
            import clip  # type: ignore
            from clip import clip as clip_impl  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install CAIDBench with `pip install -e .[clip]` to use method.name=prompt2guard. "
                "Prompt2Guard uses the official OpenAI CLIP wrapper, not OpenCLIP."
            ) from e

        official_name = self._official_clip_name(model_name)
        pretrained_path = Path(pretrained).expanduser() if isinstance(pretrained, str) else None
        local_model_path = Path(str(model_name)).expanduser()
        if pretrained not in {None, False, "openai"} and pretrained_path is not None and pretrained_path.exists():
            model_path = str(pretrained_path)
        elif official_name in clip_impl._MODELS:
            model_path = self._download_clip_model(clip_impl, official_name)
        elif local_model_path.exists():
            model_path = str(local_model_path)
        else:
            raise ValueError(f"Unknown OpenAI CLIP model {model_name!r}; available models: {clip.available_models()}")

        try:
            jit_model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = jit_model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        self.clip_model = clip_impl.build_model(state_dict)
        if str(precision).lower() == "fp32":
            self.clip_model.float()
        self.tokenizer = clip.tokenize
        self.model_name = official_name
        self.precision = str(precision).lower()
        self.K = int(k)
        self.topk_classes = int(topk_classes)
        if self.topk_classes > 5:
            raise ValueError("Official Prompt2Guard supports at most topk_classes=5.")
        self.class_names = tuple(str(x) for x in class_names)
        if len(self.class_names) != 2:
            raise ValueError("Prompt2Guard expects exactly two class names: real and fake.")

        if self.topk_classes > 1:
            flags = tuple(bool(x) for x in ensembling)
            if len(flags) != 4:
                raise ValueError("ensembling must contain four booleans.")
            (
                self.ensemble_token_embedding,
                self.ensemble_before_cosine_sim,
                self.ensemble_after_cosine_sim,
                self.confidence_score_enable,
            ) = flags
        else:
            self.ensemble_token_embedding = False
            self.ensemble_before_cosine_sim = False
            self.ensemble_after_cosine_sim = False
            self.confidence_score_enable = False

        if self.topk_classes > 1 and not any(
            (self.ensemble_token_embedding, self.ensemble_before_cosine_sim, self.ensemble_after_cosine_sim)
        ):
            raise ValueError("topk_classes > 1 requires one official object-label ensembling mode.")

        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        self.clip_model.eval()

        self.prompt_learner = nn.ModuleList()
        self.score_weights_labels: torch.Tensor | None = None
        self.text_tokenized: torch.Tensor | None = None
        self.text_x: torch.Tensor | None = None
        self.len_prompts: torch.Tensor | None = None
        self.text_mask: torch.Tensor | None = None
        self.visual_mask: torch.Tensor | None = None

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
    def _download_clip_model(clip_impl: Any, model_name: str) -> str:
        url = clip_impl._MODELS[model_name]
        download = clip_impl._download
        try:
            params = inspect.signature(download).parameters
        except (TypeError, ValueError):
            params = {}
        if "root" in params:
            return str(download(url, root=str(Path.home() / ".cache" / "clip")))
        return str(download(url))

    @staticmethod
    def _set_transformer_mask(transformer: nn.Module, attn_mask: torch.Tensor | None) -> None:
        if hasattr(transformer, "resblocks"):
            for block in transformer.resblocks:
                if hasattr(block, "attn_mask"):
                    block.attn_mask = attn_mask

    @classmethod
    def _forward_transformer(cls, transformer: nn.Module, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        try:
            return transformer(x, attn_mask)
        except TypeError:
            cls._set_transformer_mask(transformer, attn_mask)
            return transformer(x)

    @property
    def feature_dim(self) -> int:
        return int(getattr(self.clip_model.visual, "output_dim", self.clip_model.text_projection.shape[-1]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return _clip_model_dtype(self.clip_model)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.clip_model.eval()
        return self

    def add_task(self) -> int:
        self.prompt_learner.append(PromptLearner(self.clip_model, self.K))
        return len(self.prompt_learner) - 1

    def freeze_except(self, task_index: int) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.prompt_learner[task_index].parameters():
            p.requires_grad_(True)

    def make_prompts(self, prompts: list[str]) -> None:
        device = self.device
        with torch.no_grad():
            tokenized = self.tokenizer(prompts).to(device)
            text_x = self.clip_model.token_embedding(tokenized).type(self.dtype)
            text_x = text_x + self.clip_model.positional_embedding.to(device=device, dtype=self.dtype)
            self.text_tokenized = tokenized
            self.text_x = text_x
            self.len_prompts = tokenized.argmax(dim=-1) + 1

    def define_mask(self, prompt_count: int) -> None:
        if self.len_prompts is None:
            raise RuntimeError("make_prompts must run before define_mask.")
        context_len = int(self.clip_model.positional_embedding.shape[0])
        heads = int(getattr(self.clip_model.transformer.resblocks[0].attn, "num_heads", 8))
        device = self.device
        dtype = self.dtype

        max_idx = int((self.len_prompts.max() + prompt_count - 1).item())
        if max_idx >= context_len:
            raise ValueError(
                f"Prompt2Guard prompt insertion exceeds CLIP context length: index={max_idx}, context={context_len}."
            )

        text_mask = torch.full((len(self.len_prompts) * heads, context_len, context_len), float("-inf"), device=device)
        for i, idx in enumerate(self.len_prompts.tolist()):
            mask = torch.full((context_len, context_len), float("-inf"), device=device)
            mask.triu_(1)
            mask[:, int(idx) :].fill_(float("-inf"))
            text_mask[i * heads : (i + 1) * heads] = mask
        self.text_mask = text_mask.to(dtype=dtype)

        visual_tokens = int(self.clip_model.visual.positional_embedding.shape[0]) + int(prompt_count)
        visual_mask = torch.zeros((visual_tokens, visual_tokens), dtype=dtype, device=device)
        visual_mask[:, -int(prompt_count) :] = float("-inf")
        self.visual_mask = visual_mask

    @staticmethod
    def _coerce_one_object_list(value: Any) -> list[tuple[str, float]]:
        if isinstance(value, Mapping):
            parsed = parse_object_labels({"object_labels": value})
            return parsed or []
        if isinstance(value, str):
            parsed = parse_object_labels({"object_labels": value})
            return parsed or []
        if isinstance(value, tuple) and len(value) >= 1 and not isinstance(value[0], (list, tuple, Mapping)):
            label = str(value[0])
            score = float(value[1]) if len(value) > 1 else 1.0
            return [(label, score)]
        if isinstance(value, list):
            out: list[tuple[str, float]] = []
            for item in value:
                out.extend(SliNet._coerce_one_object_list(item))
            return out
        return [(str(value), 1.0)]

    def _batch_object_labels(self, object_labels: Any, batch_size: int) -> tuple[list[list[str]], torch.Tensor]:
        if object_labels is None:
            if self.topk_classes > 0:
                raise ValueError(
                    "Prompt2Guard requires batch['object_labels'] when topk_classes > 0. "
                    "Add an object_labels/topk_object_labels column to Arrow metadata."
                )
            return [[] for _ in range(batch_size)], torch.ones(batch_size, 0, device=self.device)

        if not isinstance(object_labels, list) or (object_labels and not isinstance(object_labels[0], (list, tuple, Mapping, str))):
            object_labels = [object_labels]
        if len(object_labels) != batch_size:
            raise ValueError(f"Prompt2Guard expected {batch_size} object-label entries, got {len(object_labels)}.")

        labels_by_sample: list[list[str]] = []
        scores_by_sample: list[list[float]] = []
        for item in object_labels:
            pairs = self._coerce_one_object_list(item)
            if len(pairs) < self.topk_classes:
                raise ValueError(
                    f"Prompt2Guard needs at least topk_classes={self.topk_classes} object labels per sample; got {len(pairs)}."
                )
            pairs = pairs[: self.topk_classes]
            labels_by_sample.append([label for label, _score in pairs])
            scores_by_sample.append([float(score) for _label, score in pairs])

        scores = torch.tensor(scores_by_sample, dtype=torch.float32, device=self.device)
        if scores.numel() and float(scores.max()) > 1.0:
            scores = scores / 100.0
        return labels_by_sample, scores

    def generate_prompts_from_input(self, object_labels: Any, batch_size: int) -> None:
        labels_by_sample, scores = self._batch_object_labels(object_labels, batch_size)

        if self.confidence_score_enable and self.topk_classes > 0:
            weights = scores[:, : self.topk_classes].unsqueeze(1).expand(-1, 2, -1)
            self.score_weights_labels = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            self.score_weights_labels = None

        if self.topk_classes > 0:
            if self.topk_classes == 1:
                prompts = [
                    f"a {type_image} photo of a {labels[0]}."
                    for labels in labels_by_sample
                    for type_image in self.class_names
                ]
                self.make_prompts(prompts)
            else:
                prompts = [
                    f"a {type_image} photo of a {obj_label}."
                    for labels in labels_by_sample
                    for type_image in self.class_names
                    for obj_label in labels[: self.topk_classes]
                ]
                if self.ensemble_token_embedding:
                    self.make_prompts(prompts)
                    if self.text_x is None or self.text_tokenized is None:
                        raise RuntimeError("Prompt text state was not initialized.")
                    self.len_prompts = torch.cat(
                        [
                            self.text_tokenized[i : i + self.topk_classes].argmax(dim=-1).max().unsqueeze(0) + 1
                            for i in range(0, len(self.text_tokenized), self.topk_classes)
                        ]
                    )
                    text_x = self.text_x.reshape(batch_size, 2, self.topk_classes, *self.text_x.shape[1:])
                    self.text_x = text_x.mean(dim=2).reshape(batch_size * 2, *self.text_x.shape[1:])
                else:
                    self.make_prompts(prompts)
        else:
            prompts = [
                f"a photo of a {type_image} image."
                for _ in range(batch_size)
                for type_image in self.class_names
            ]
            self.make_prompts(prompts)

    def text_encoder(self, text_prompt: torch.Tensor) -> torch.Tensor:
        if self.text_x is None or self.len_prompts is None:
            raise RuntimeError("generate_prompts_from_input must run before text_encoder.")
        prompt_count = int(text_prompt.shape[0])
        self.define_mask(prompt_count)
        if self.text_mask is None:
            raise RuntimeError("Text attention mask was not initialized.")

        text_x = self.text_x.to(device=self.device, dtype=self.dtype).clone()
        text_prompt = text_prompt.to(device=self.device, dtype=self.dtype)
        rows = torch.arange(text_x.shape[0], device=self.device)
        for i in range(prompt_count):
            text_x[rows, self.len_prompts.to(self.device) + i, :] = text_prompt[i, :].repeat(text_x.shape[0], 1)

        text_x = text_x.permute(1, 0, 2)
        text_x = self._forward_transformer(self.clip_model.transformer, text_x, self.text_mask)
        text_x = text_x.permute(1, 0, 2)
        text_x = self.clip_model.ln_final(text_x).type(self.dtype)

        features = []
        for i in range(prompt_count):
            idx = self.len_prompts.to(self.device) + i
            features.append(text_x[rows, idx, :])
        text_f = torch.stack(features, dim=1)
        text_f = text_f @ self.clip_model.text_projection.to(device=self.device, dtype=self.dtype)

        if self.ensemble_before_cosine_sim:
            batch_size = self.text_x.shape[0] // (2 * self.topk_classes)
            text_f = text_f.reshape(batch_size, 2, self.topk_classes, prompt_count, -1).mean(dim=2)
            text_f = text_f.reshape(batch_size * 2, prompt_count, -1)
        return text_f

    def image_encoder(self, image: torch.Tensor, image_prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.visual_mask is None:
            self.define_mask(int(image_prompt.shape[0] if image_prompt.dim() == 2 else image_prompt.shape[1]))
        if self.visual_mask is None:
            raise RuntimeError("Visual attention mask was not initialized.")

        batch_size = image.shape[0]
        image = image.to(device=self.device, dtype=self.dtype)
        image_prompt = image_prompt.to(device=self.device, dtype=self.dtype)
        if image_prompt.dim() == 2:
            image_prompt = image_prompt.unsqueeze(0).repeat(batch_size, 1, 1)

        x = self.clip_model.visual.conv1(image)
        x = x.reshape(batch_size, x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype).view(1, 1, -1).repeat(batch_size, 1, 1), x], dim=1)
        if x.shape[1] != self.clip_model.visual.positional_embedding.shape[0]:
            raise ValueError(
                "Prompt2Guard official CLIP path expects 224x224 images for ViT-B/16. "
                "Use a YAML transform ending in a 224x224 crop or resize."
            )
        x = x + self.clip_model.visual.positional_embedding.to(device=self.device, dtype=self.dtype)
        x = torch.cat([x, image_prompt], dim=1)
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self._forward_transformer(self.clip_model.visual.transformer, x, self.visual_mask)
        x = x.permute(1, 0, 2)
        prompt_tokens = self.clip_model.visual.ln_post(x[:, -image_prompt.shape[1] :, :])
        image_cls = self.clip_model.visual.ln_post(x[:, 0, :])
        visual_proj = self.clip_model.visual.proj
        if visual_proj is not None:
            visual_proj = visual_proj.to(device=self.device, dtype=self.dtype)
            prompt_tokens = prompt_tokens @ visual_proj
            image_cls = image_cls @ visual_proj
        return prompt_tokens, image_cls

    def extract_vector(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self._set_transformer_mask(self.clip_model.visual.transformer, None)
            features = self.clip_model.encode_image(image.to(device=self.device, dtype=self.dtype))
            features = F.normalize(features, dim=-1)
        return features.float()

    def forward(self, image: torch.Tensor, object_labels: Any) -> dict[str, torch.Tensor]:
        if not self.prompt_learner:
            raise RuntimeError("Prompt2Guard has no prompt learner; call before_task first.")
        text_prompt, image_prompt = self.prompt_learner[-1]()
        self.generate_prompts_from_input(object_labels, batch_size=image.shape[0])
        text_f = F.normalize(self.text_encoder(text_prompt), dim=-1)
        img_f, _ = self.image_encoder(image, image_prompt)
        img_f = F.normalize(img_f, dim=-1)
        logits = self.training_cosine_similarity(text_f, img_f)
        return {"logits": logits}

    def training_cosine_similarity(self, text_f: torch.Tensor, img_f: torch.Tensor) -> torch.Tensor:
        if self.ensemble_after_cosine_sim:
            text_f = text_f.reshape(img_f.shape[0], 2, self.topk_classes, self.K, -1)
            logits = torch.zeros(img_f.shape[0], 2, device=self.device, dtype=img_f.dtype)
            for i in range(self.K):
                logit = torch.einsum("bd,blod->blo", img_f[:, i, :], text_f[:, :, :, i, :])
                if self.confidence_score_enable and self.score_weights_labels is not None:
                    logit = torch.einsum("blo,blo->bl", logit, self.score_weights_labels.to(logit.dtype))
                else:
                    logit = logit.mean(dim=-1)
                logits += self.clip_model.logit_scale.exp().to(logit.dtype) * logit
            return logits / self.K

        text_f = text_f.reshape(img_f.shape[0], 2, self.K, -1)
        logits = torch.zeros(img_f.shape[0], 2, device=self.device, dtype=img_f.dtype)
        for i in range(self.K):
            logits += self.clip_model.logit_scale.exp().to(img_f.dtype) * torch.einsum("bd,bpd->bp", img_f[:, i, :], text_f[:, :, i, :])
        return logits / self.K

    def convert_to_prob_distribution(self, keys: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        keys = keys.to(device=self.device, dtype=image_features.dtype)
        if keys.dim() == 1:
            keys = keys.unsqueeze(0)
        return F.softmax(torch.einsum("bd,td->bt", image_features, keys), dim=1)

    def interface(self, image: torch.Tensor, object_labels: Any, keys_dict: dict[str, torch.Tensor], prototype: str = "fake") -> torch.Tensor:
        total_tasks = len(self.prompt_learner)
        if total_tasks == 0:
            raise RuntimeError("Prompt2Guard has no prompt learners for inference.")
        img_prompts = torch.cat([learner.img_prompt for learner in self.prompt_learner], dim=0)
        text_prompts = torch.cat([learner.text_prompt for learner in self.prompt_learner], dim=0)
        self.generate_prompts_from_input(object_labels, batch_size=image.shape[0])
        text_f = F.normalize(self.text_encoder(text_prompts), dim=-1)
        img_f, image_features = self.image_encoder(image, img_prompts)
        img_f = F.normalize(img_f, dim=-1)

        prob_dist = {
            "real": self.convert_to_prob_distribution(keys_dict["real_keys_one_cluster"], image_features),
            "fake": self.convert_to_prob_distribution(keys_dict["fake_keys_one_cluster"], image_features),
            "all": self.convert_to_prob_distribution(keys_dict["all_keys_one_cluster"], image_features),
        }
        if prototype not in prob_dist:
            raise KeyError(f"Unknown Prompt2Guard prototype={prototype!r}; use real, fake, or all.")
        return self.inference_cosine_similarity(text_f, img_f, prob_dist[prototype], total_tasks)

    def inference_cosine_similarity(
        self,
        text_f: torch.Tensor,
        img_f: torch.Tensor,
        task_prob: torch.Tensor,
        total_tasks: int,
    ) -> torch.Tensor:
        if self.ensemble_after_cosine_sim:
            text_f = text_f.reshape(img_f.shape[0], 2, self.topk_classes, self.K * total_tasks, -1)
            logits = []
            for t in range(total_tasks):
                logits_tmp = torch.zeros(img_f.shape[0], 2, device=self.device, dtype=img_f.dtype)
                # Official Prompt2Guard weights both image and text prompt features with the selected task probability.
                image_weight = task_prob[:, t].unsqueeze(-1).to(img_f.dtype)
                text_weight = image_weight.unsqueeze(-1).unsqueeze(-1)
                for k in range(self.K):
                    offset = k + t * self.K
                    logit = torch.einsum(
                        "bd,blod->blo",
                        img_f[:, offset, :] * image_weight,
                        text_f[:, :, :, offset, :] * text_weight,
                    )
                    if self.confidence_score_enable and self.score_weights_labels is not None:
                        logit = torch.einsum("blo,blo->bl", logit, self.score_weights_labels.to(logit.dtype))
                    else:
                        logit = logit.mean(dim=-1)
                    logits_tmp += self.clip_model.logit_scale.exp().to(logit.dtype) * logit
                logits.append(logits_tmp / self.K)
            return torch.stack(logits, dim=1)

        text_f = text_f.reshape(img_f.shape[0], 2, self.K * total_tasks, -1)
        logits = []
        for t in range(total_tasks):
            logits_tmp = torch.zeros(img_f.shape[0], 2, device=self.device, dtype=img_f.dtype)
            # Official Prompt2Guard weights both image and text prompt features with the selected task probability.
            image_weight = task_prob[:, t].unsqueeze(-1).to(img_f.dtype)
            text_weight = image_weight.unsqueeze(-1)
            for k in range(self.K):
                offset = k + t * self.K
                logit = torch.einsum(
                    "bd,bpd->bp",
                    img_f[:, offset, :] * image_weight,
                    text_f[:, :, offset, :] * text_weight,
                )
                logits_tmp += self.clip_model.logit_scale.exp().to(logit.dtype) * logit
            logits.append(logits_tmp / self.K)
        return torch.stack(logits, dim=1)


@register_method("prompt2guard")
class Prompt2GuardMethod(ContinualMethod):
    """Official Prompt2Guard implementation for continual real/fake detection."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str | bool | None = "openai",
        K: int = 7,
        topk_classes: int = 5,
        ensembling: tuple[bool, bool, bool, bool] | list[bool] = (False, False, True, False),
        class_names: tuple[str, str] | list[str] = ("real", "fake"),
        precision: str = "fp16",
        prototype: str = "fake",
        prediction_mode: str = "mix_top_mean",
        n_clusters: int = 5,
        label_smoothing: float = 0.1,
        enable_prev_prompt: bool = False,
        object_label_sidecar: str | None = None,
        object_label_root: str | None = None,
        init_lr: float = 0.01,
        lrate: float = 0.01,
        init_weight_decay: float = 5e-4,
        weight_decay: float = 2e-4,
        warmup_epoch: int = 1,
        warmup_lr: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        self.num_classes = 2
        self.current_task_id: int | None = None
        self.extra_cfg = dict(kwargs)
        self.network = SliNet(
            model_name=model_name,
            pretrained=pretrained,
            k=K,
            topk_classes=topk_classes,
            ensembling=ensembling,
            class_names=class_names,
            precision=precision,
        )
        self.prototype = str(prototype).lower()
        self.prediction_mode = str(prediction_mode).lower()
        self.n_clusters = int(n_clusters)
        self.label_smoothing = float(label_smoothing)
        self.enable_prev_prompt = bool(enable_prev_prompt)
        self.object_label_root = Path(object_label_root).expanduser().resolve() if object_label_root else None
        self.object_label_lookup = self._load_object_label_sidecar(object_label_sidecar)
        self.init_lr = float(init_lr)
        self.lrate = float(lrate)
        self.init_weight_decay = float(init_weight_decay)
        self.weight_decay = float(weight_decay)
        self.warmup_epoch = int(warmup_epoch)
        self.warmup_lr = float(warmup_lr)
        self.task_ids: list[int] = []
        self.current_task_index = -1

    def _load_object_label_sidecar(self, path: str | None) -> dict[str, Any]:
        if not path:
            return {}
        sidecar = Path(path).expanduser()
        with open(sidecar, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Prompt2Guard object_label_sidecar must contain a mapping, got {type(payload)!r}.")
        return {str(k): v for k, v in payload.items()}

    def _path_lookup_keys(self, path_value: Any) -> list[str]:
        raw = str(path_value)
        keys = [raw]
        p = Path(raw)
        if self.object_label_root is not None:
            try:
                rel = p.resolve().relative_to(self.object_label_root)
                keys.extend([str(rel), "/" + str(rel)])
            except Exception:
                pass
        keys.extend([p.name, "/" + p.name])
        deduped: list[str] = []
        for key in keys:
            normalized = key.replace("\\", "/")
            if normalized not in deduped:
                deduped.append(normalized)
        return deduped

    def _object_labels_for_batch(self, batch: dict[str, Any]) -> Any:
        if batch.get("object_labels") is not None:
            return batch["object_labels"]
        if not self.object_label_lookup:
            return None
        paths = batch.get("path")
        if paths is None:
            return None
        if isinstance(paths, (str, Path)):
            paths = [paths]
        out = []
        missing = []
        for path in paths:
            found = None
            for key in self._path_lookup_keys(path):
                if key in self.object_label_lookup:
                    found = self.object_label_lookup[key]
                    break
            if found is None:
                missing.append(str(path))
            else:
                out.append(found)
        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(
                f"Prompt2Guard object_label_sidecar has no entries for {len(missing)} batch paths, e.g. {preview}. "
                "Set method.object_label_root to the dataset root used by the official classes.pkl keys."
            )
        return out

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def before_task(self, task: Any, train_loader: Any | None = None) -> None:
        self.current_task_id = int(getattr(task, "task_id", task if isinstance(task, int) else len(self.task_ids)))
        if self.current_task_id not in self.task_ids:
            self.task_ids.append(self.current_task_id)
            new_task_index = self.network.add_task()
            if self.enable_prev_prompt and new_task_index > 0:
                self.network.prompt_learner[new_task_index].load_state_dict(
                    self.network.prompt_learner[new_task_index - 1].state_dict()
                )
        self.current_task_index = self.task_ids.index(self.current_task_id)
        self.network.freeze_except(self.current_task_index)

    def configure_optimizer(self, optimizer_cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
        cfg = dict(optimizer_cfg or {})
        cfg.setdefault("type", "sgd")
        cfg.setdefault("momentum", 0.9)
        if "lr" not in cfg:
            cfg["lr"] = self.init_lr if self.current_task_index == 0 else self.lrate
        if "weight_decay" not in cfg:
            cfg["weight_decay"] = self.init_weight_decay if self.current_task_index == 0 else self.weight_decay
        return build_optimizer(self.parameters(), cfg)

    def _set_epoch_lr(self, optimizer: torch.optim.Optimizer, base_lrs: list[float], epoch: int, epochs: int) -> None:
        if self.warmup_epoch > 0 and epoch < self.warmup_epoch:
            lrs = [self.warmup_lr for _ in base_lrs]
        else:
            denom = max(epochs - self.warmup_epoch, 1)
            progress = min(max((epoch - self.warmup_epoch) / denom, 0.0), 1.0)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs = [base_lr * factor for base_lr in base_lrs]
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr

    def fit_task(self, trainer: Any, task: Any, train_loader: Any, val_loader: Any | None = None) -> bool:
        self.train()
        optimizer = self.configure_optimizer(trainer.optimizer_cfg)
        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        epochs = int(trainer.max_epochs)
        for epoch in range(epochs):
            self._set_epoch_lr(optimizer, base_lrs, epoch, epochs)
            total_loss = 0.0
            total = 0
            correct = 0
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].long().to(self.device)
                out = self.network(x, self._object_labels_for_batch(batch))
                loss = F.cross_entropy(out["logits"], y, label_smoothing=self.label_smoothing)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if trainer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), trainer.grad_clip)
                optimizer.step()
                trainer.advance_step()
                total_loss += float(loss.detach().cpu())
                pred = out["logits"].argmax(dim=1)
                correct += int((pred == y).sum().detach().cpu())
                total += int(y.numel())
            train_metrics = {
                "train/prompt2guard_loss": total_loss / max(len(train_loader), 1),
                "train/prompt2guard_acc": correct / max(total, 1),
                "train/task_index": float(getattr(task, "task_id", 0)),
                "train/epoch": epoch + 1,
            }
            trainer.logger.info(
                "task=%s epoch=%d/%d prompt2guard_loss=%.4f prompt2guard_acc=%.4f",
                task.name,
                epoch + 1,
                epochs,
                train_metrics["train/prompt2guard_loss"],
                train_metrics["train/prompt2guard_acc"],
            )
            trainer.log_metrics(train_metrics)
        return True

    def _cluster_features(self, features: torch.Tensor, n_clusters: int) -> torch.Tensor:
        features = F.normalize(features.float(), dim=-1).detach().cpu()
        if features.numel() == 0:
            return features.reshape(0, self.network.feature_dim)
        values = features.numpy()
        unique = np.unique(values, axis=0)
        n_clusters = min(max(int(n_clusters), 1), len(unique))
        if n_clusters == 1:
            centers = values.mean(axis=0, keepdims=True)
        else:
            try:
                km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(values)
            except TypeError:  # scikit-learn < 1.4
                km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(values)
            centers = km.cluster_centers_
        return torch.as_tensor(centers, dtype=torch.float32)

    def _buffer_name(self, kind: str, task_index: int) -> str:
        return f"prompt2guard_{kind}_{task_index}"

    def _store_key(self, kind: str, task_index: int, value: torch.Tensor) -> None:
        name = self._buffer_name(kind, task_index)
        value = value.detach().to(self.device)
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is None:
            return
        all_features: list[torch.Tensor] = []
        real_features: list[torch.Tensor] = []
        fake_features: list[torch.Tensor] = []
        self.eval()
        for batch in train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].long()
            z = self.network.extract_vector(x).detach().cpu()
            all_features.append(z)
            real_features.append(z[y == 0])
            fake_features.append(z[y == 1])
        all_z = torch.cat(all_features, dim=0) if all_features else torch.empty(0, self.network.feature_dim)
        real_z = torch.cat(real_features, dim=0) if real_features else all_z
        fake_z = torch.cat(fake_features, dim=0) if fake_features else all_z
        if real_z.numel() == 0:
            real_z = all_z
        if fake_z.numel() == 0:
            fake_z = all_z
        idx = self.current_task_index
        self._store_key("all_keys", idx, self._cluster_features(all_z, self.n_clusters))
        self._store_key("all_keys_one_cluster", idx, self._cluster_features(all_z, 1).squeeze(0))
        self._store_key("real_keys_one_cluster", idx, self._cluster_features(real_z, 1).squeeze(0))
        self._store_key("fake_keys_one_cluster", idx, self._cluster_features(fake_z, 1).squeeze(0))
        self.network.freeze_except(self.current_task_index)

    def _stack_keys(self, kind: str) -> torch.Tensor:
        tensors = []
        for idx in range(len(self.task_ids)):
            name = self._buffer_name(kind, idx)
            if name not in self._buffers:
                raise RuntimeError(f"Prompt2Guard missing prototype buffer {name}; after_task must run before evaluation.")
            tensors.append(getattr(self, name).reshape(-1, self.network.feature_dim)[0])
        return torch.stack(tensors, dim=0)

    def _keys_dict(self) -> dict[str, torch.Tensor]:
        return {
            "all_keys_one_cluster": self._stack_keys("all_keys_one_cluster"),
            "real_keys_one_cluster": self._stack_keys("real_keys_one_cluster"),
            "fake_keys_one_cluster": self._stack_keys("fake_keys_one_cluster"),
        }

    def _aggregate_logits(self, outputs: torch.Tensor) -> torch.Tensor:
        mode = self.prediction_mode
        if mode == "mean":
            return outputs.mean(dim=1)
        if mode == "top1":
            return outputs.max(dim=1).values
        if mode in {"mix_top_mean", "mean_max", "max_mean"}:
            per_class_max = outputs.max(dim=1).values
            per_class_mean = outputs.mean(dim=1)
            diff_max = torch.abs(per_class_max[:, 0] - per_class_max[:, 1])
            diff_mean = torch.abs(per_class_mean[:, 0] - per_class_mean[:, 1])
            use_mean = diff_mean > diff_max
            return torch.where(use_mean[:, None], per_class_mean, per_class_max)
        raise KeyError(f"Unknown Prompt2Guard prediction_mode={self.prediction_mode!r}.")

    def predict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        outputs = self.network.interface(x, self._object_labels_for_batch(batch), self._keys_dict(), prototype=self.prototype)
        logits = self._aggregate_logits(outputs)
        return {"logits": logits, "task_logits": outputs}

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        x = batch["x"].to(self.device)
        y = batch["y"].long().to(self.device)
        out = self.network(x, self._object_labels_for_batch(batch))
        loss = F.cross_entropy(out["logits"], y, label_smoothing=self.label_smoothing)
        return {"loss": loss, "ce": loss.detach()}
