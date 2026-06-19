from __future__ import annotations

import sys
import types

import torch
from torch import nn

from caidbench.methods.sprompts import PromptedOpenCLIPVisionEncoder


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
