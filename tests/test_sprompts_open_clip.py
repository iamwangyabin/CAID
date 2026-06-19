from __future__ import annotations

import sys
import types

import torch
from torch import nn

from caidbench.methods.sprompts import PromptedOpenCLIPVisionEncoder, SPromptsMethod


class _SequenceFirstTransformer(nn.Module):
    batch_first = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 2
        return x


class _OpenClipVisualWithoutPool(nn.Module):
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=2, stride=2, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(8))
        self.positional_embedding = nn.Parameter(torch.zeros(17, 8))
        self.ln_pre = nn.LayerNorm(8)
        self.transformer = _SequenceFirstTransformer()
        self.ln_post = nn.LayerNorm(8)
        self.proj = nn.Parameter(torch.randn(8, self.output_dim))
        self.patch_dropout = nn.Identity()


class _OpenClipModelWithoutPool(nn.Module):
    embed_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.visual = _OpenClipVisualWithoutPool()


def test_open_clip_encoder_supports_visual_without_pool(monkeypatch) -> None:
    clip_model = _OpenClipModelWithoutPool()
    open_clip_stub = types.SimpleNamespace(create_model=lambda *args, **kwargs: clip_model)
    monkeypatch.setitem(sys.modules, "open_clip", open_clip_stub)

    encoder = PromptedOpenCLIPVisionEncoder({"type": "open_clip", "model_name": "ViT-B-16", "pretrained": None})
    out = encoder(torch.randn(2, 3, 8, 8), torch.randn(3, encoder.prompt_dim))

    assert out.shape == (2, 4)


class _SequenceFirstTextTransformer(nn.Module):
    batch_first = False

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def get_cast_dtype(self) -> torch.dtype:
        return torch.float32

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.shape == (77, 2, self.dim)
        assert attn_mask is not None
        assert attn_mask.shape == (77, 77)
        return x


class _TextClipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _SequenceFirstTextTransformer(dim=8)
        self.positional_embedding = nn.Parameter(torch.zeros(77, 8))
        self.ln_final = nn.LayerNorm(8)
        self.text_projection = nn.Parameter(torch.randn(8, 4))
        self.register_buffer("attn_mask", torch.zeros(77, 77))
        self.text_pool_type = "argmax"
        self.text_eos_id = None


class _TextLearner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("tokenized_prompts", torch.tensor([[0] * 76 + [1], [0] * 76 + [1]], dtype=torch.long))

    def forward(self) -> torch.Tensor:
        return torch.randn(2, 77, 8)


def test_text_prompt_encoder_supports_sequence_first_transformer(monkeypatch) -> None:
    transformer_stub = types.ModuleType("open_clip.transformer")

    def text_global_pool(
        x: torch.Tensor,
        tokenized: torch.Tensor,
        pool_type: str,
    ) -> torch.Tensor:
        assert x.shape == (2, 77, 8)
        return x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)]

    transformer_stub.text_global_pool = text_global_pool
    monkeypatch.setitem(sys.modules, "open_clip.transformer", transformer_stub)

    method = SPromptsMethod.__new__(SPromptsMethod)
    method.image_encoder = types.SimpleNamespace(clip_model=_TextClipModel())

    out = method._encode_text_prompts(_TextLearner())

    assert out.shape == (2, 4)
