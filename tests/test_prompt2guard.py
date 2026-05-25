from __future__ import annotations

from types import SimpleNamespace

import torch

from caidbench.methods.prompt2guard import Prompt2GuardMethod
from caidbench.methods.prompt2guard import SliNet


def test_prompt2guard_normalizes_openai_clip_model_name() -> None:
    assert SliNet._official_clip_name("ViT-B-16") == "ViT-B/16"


def test_prompt2guard_one_cluster_prototype_uses_sample_mean() -> None:
    method = Prompt2GuardMethod.__new__(Prompt2GuardMethod)
    method.network = SimpleNamespace(feature_dim=2)

    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    center = method._cluster_features(features, n_clusters=1)

    assert torch.allclose(center, torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]))
    assert not torch.allclose(center.norm(dim=-1), torch.ones(1))
