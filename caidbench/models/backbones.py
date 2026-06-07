from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .e3_official import MISLNetBackbone


class SmallConvBackbone(nn.Module):
    """CPU-safe image backbone for smoke tests and small experiments.

    Production reproduction should swap this for the paper backbone: CLIP ViT,
    Xception, EfficientNet, MISLnet, SR-Net, etc.
    """

    def __init__(self, out_dim: int = 512, in_channels: int = 3) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x.float()).flatten(1)
        return self.proj(z)


class TimmBackbone(nn.Module):
    """Optional timm wrapper. Import is lazy so CAIDBench works without timm."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_dim: int | None = None,
        drop_rate: float = 0.0,
        pretrained_cfg_overlay: dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Install timm to use TimmBackbone") from e
        create_kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
        if pretrained_cfg_overlay:
            create_kwargs["pretrained_cfg_overlay"] = pretrained_cfg_overlay
        try:
            self.model = timm.create_model(model_name, **create_kwargs)
        except TypeError as e:
            if pretrained_cfg_overlay:
                raise TypeError(
                    "Installed timm does not support pretrained_cfg_overlay; upgrade timm "
                    "or set backbone.checkpoint_path to an official local checkpoint."
                ) from e
            raise
        if checkpoint_path:
            try:
                from timm.models import load_checkpoint  # type: ignore
            except Exception as e:  # pragma: no cover - optional dependency
                raise ImportError("Installed timm does not expose timm.models.load_checkpoint") from e
            load_checkpoint(self.model, checkpoint_path, strict=False)
        dim = getattr(self.model, "num_features", None)
        if dim is None:
            raise ValueError(f"Cannot infer feature dimension for timm model {model_name}")
        self.out_dim = int(out_dim or dim)
        self.dropout = nn.Dropout(float(drop_rate))
        self.proj = nn.Identity() if self.out_dim == dim else nn.Linear(dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(self.model(x)))


class CLIPVisionBackbone(nn.Module):
    """Online CLIP image feature extractor.

    This wrapper intentionally computes CLIP features from raw image tensors
    inside the model forward pass.

    Supported backends:
      - ``open_clip``: ``pip install open_clip_torch``; default for ViT-L/14.
      - ``transformers``: ``pip install transformers``; e.g. ``openai/clip-vit-large-patch14``.

    For the closest HSIC-style run, keep CLIP frozen and use either the global
    image embedding or a named intermediate module via ``hook_module`` when the
    chosen backend exposes that module.
    """

    def __init__(
        self,
        backend: str = "open_clip",
        model_name: str = "ViT-L-14",
        pretrained: str | bool = "openai",
        freeze: bool = True,
        out_dim: int | None = None,
        normalize: bool = False,
        hook_module: str | None = None,
        token_pool: str = "cls",
        hidden_layer: int = -1,
    ) -> None:
        super().__init__()
        self.backend = str(backend).lower()
        self.model_name = str(model_name)
        self.normalize = bool(normalize)
        self.hook_module = hook_module
        self.token_pool = str(token_pool).lower()
        self.hidden_layer = int(hidden_layer)

        if self.backend == "open_clip":
            self._init_open_clip(pretrained=pretrained, out_dim=out_dim)
        elif self.backend == "transformers":
            self._init_transformers(out_dim=out_dim)
        else:
            raise KeyError(f"Unknown CLIP backend: {backend}. Use 'open_clip' or 'transformers'.")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def _init_open_clip(self, pretrained: str | bool, out_dim: int | None) -> None:
        try:
            import open_clip  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Install CAIDBench with `pip install -e .[clip]` or install open_clip_torch to use backbone.type=clip_vision") from e
        pretrained_name = pretrained if isinstance(pretrained, str) else ("openai" if pretrained else None)
        self.model = open_clip.create_model(self.model_name, pretrained=pretrained_name)
        visual = getattr(self.model, "visual", self.model)
        inferred = getattr(visual, "output_dim", None) or getattr(visual, "embed_dim", None)
        self.out_dim = int(out_dim or inferred or 768)
        if out_dim is not None and inferred is not None and int(out_dim) != int(inferred) and self.hook_module is None:
            self.proj = nn.Linear(int(inferred), int(out_dim))
        else:
            self.proj = nn.Identity()

    def _init_transformers(self, out_dim: int | None) -> None:
        try:
            from transformers import CLIPVisionModel  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Install CAIDBench with `pip install -e .[clip]` or install transformers to use backend=transformers") from e
        self.model = CLIPVisionModel.from_pretrained(self.model_name)
        hidden = int(getattr(self.model.config, "hidden_size", out_dim or 768))
        proj_dim = int(out_dim or hidden)
        self.out_dim = proj_dim
        self.proj = nn.Identity() if proj_dim == hidden else nn.Linear(hidden, proj_dim)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if all(not p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self

    @staticmethod
    def _pool_activation(z: Any, token_pool: str = "cls") -> torch.Tensor:
        if isinstance(z, (tuple, list)):
            z = z[0]
        if not torch.is_tensor(z):
            raise TypeError(f"Hooked CLIP module returned non-tensor activation: {type(z)!r}")
        if z.ndim == 4:
            return F.adaptive_avg_pool2d(z.float(), 1).flatten(1)
        if z.ndim == 3:
            if token_pool == "mean":
                return z.float().mean(dim=1)
            return z[:, 0].float()
        return z.reshape(z.shape[0], -1).float()

    def _forward_open_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.hook_module:
            activations: dict[str, torch.Tensor] = {}
            module = self.model.get_submodule(self.hook_module)
            handle = module.register_forward_hook(lambda _m, _inp, out: activations.setdefault("z", out))
            try:
                _ = self.model.encode_image(x)
            finally:
                handle.remove()
            z = self._pool_activation(activations["z"], token_pool=self.token_pool)
            if z.shape[1] != self.out_dim:
                # For hooked intermediate features, users should set out_dim to
                # the pooled activation size. A dynamic projection here would be
                # created after optimizer construction, so fail loudly instead.
                raise ValueError(
                    f"hook_module={self.hook_module!r} produced dim={z.shape[1]}, but backbone.out_dim={self.out_dim}. "
                    "Set method.detector_cfg.backbone.out_dim to this activation dimension."
                )
            return z
        z = self.model.encode_image(x)
        z = z.float()
        return self.proj(z)

    def _forward_transformers(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x, output_hidden_states=True)
        if self.hook_module:
            raise ValueError("backend=transformers uses hidden_layer instead of hook_module")
        if out.hidden_states is not None:
            z = out.hidden_states[self.hidden_layer]
            z = self._pool_activation(z, token_pool=self.token_pool)
        else:
            z = out.pooler_output
        return self.proj(z.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "CLIPVisionBackbone expects raw image tensors [B,3,H,W]. "
                "Configure scenario.data with raw images, not pre-extracted feature rows."
            )
        if self.backend == "open_clip":
            z = self._forward_open_clip(x.float())
        else:
            z = self._forward_transformers(x.float())
        return F.normalize(z, dim=1) if self.normalize else z


def build_backbone(cfg: dict[str, Any] | None = None) -> nn.Module:
    cfg = cfg or {}
    kind = str(cfg.get("type", "small_conv")).lower()
    if kind in {"small_conv", "cnn", "simple_cnn"}:
        return SmallConvBackbone(out_dim=int(cfg.get("out_dim", 512)), in_channels=int(cfg.get("in_channels", 3)))
    if kind == "timm":
        return TimmBackbone(
            cfg["name"],
            pretrained=bool(cfg.get("pretrained", True)),
            out_dim=cfg.get("out_dim"),
            drop_rate=float(cfg.get("drop_rate", 0.0)),
            pretrained_cfg_overlay=cfg.get("pretrained_cfg_overlay"),
            checkpoint_path=cfg.get("checkpoint_path") or cfg.get("pretrained_path"),
        )
    if kind in {"clip", "clip_vision", "online_clip", "open_clip"}:
        return CLIPVisionBackbone(
            backend=str(cfg.get("backend", "open_clip")),
            model_name=str(cfg.get("model_name", cfg.get("name", "ViT-L-14"))),
            pretrained=cfg.get("pretrained", "openai"),
            freeze=bool(cfg.get("freeze", True)),
            out_dim=cfg.get("out_dim"),
            normalize=bool(cfg.get("normalize", False)),
            hook_module=cfg.get("hook_module"),
            token_pool=str(cfg.get("token_pool", "cls")),
            hidden_layer=int(cfg.get("hidden_layer", -1)),
        )
    if kind in {"mislnet", "misl", "e3_mislnet"}:
        return MISLNetBackbone(
            patch_size=int(cfg.get("patch_size", 256)),
            num_filters=int(cfg.get("num_filters", 6)),
            constrained_conv=bool(cfg.get("constrained_conv", True)),
            save_features=bool(cfg.get("save_features", False)),
        )
    raise KeyError(f"Unknown backbone type: {kind}")
