from __future__ import annotations

from types import SimpleNamespace

import torch

from caidbench.methods.prompt2guard import Prompt2GuardMethod
from caidbench.methods.prompt2guard import SliNet
from caidbench.methods.prompt2guard import _clip_model_dtype


class _TinyPrompt(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))


class _TinyPromptNetwork:
    def __init__(self) -> None:
        self.prompt_learner = torch.nn.ModuleList()
        self.frozen_indices: list[int] = []

    def add_task(self) -> int:
        self.prompt_learner.append(_TinyPrompt(float(len(self.prompt_learner))))
        return len(self.prompt_learner) - 1

    def freeze_except(self, task_index: int) -> None:
        self.frozen_indices.append(task_index)


class _RootRequiredClip:
    _MODELS = {"ViT-B/16": "https://example.invalid/clip.pt"}

    @staticmethod
    def _download(url: str, root: str) -> str:
        return f"{root}/{url.rsplit('/', 1)[-1]}"


def test_prompt2guard_normalizes_openai_clip_model_name() -> None:
    assert SliNet._official_clip_name("ViT-B-16") == "ViT-B/16"


def test_prompt2guard_uses_visual_dtype_over_token_embedding_dtype() -> None:
    clip_model = SimpleNamespace(
        dtype=torch.float16,
        token_embedding=SimpleNamespace(weight=torch.empty(1, dtype=torch.float32)),
        visual=SimpleNamespace(conv1=SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))),
    )

    assert _clip_model_dtype(clip_model) == torch.float16


def test_prompt2guard_downloads_openai_clip_with_cache_root() -> None:
    path = SliNet._download_clip_model(_RootRequiredClip, "ViT-B/16")

    assert path.endswith("/.cache/clip/clip.pt")


def test_prompt2guard_enable_prev_prompt_initializes_new_task_from_previous() -> None:
    method = Prompt2GuardMethod.__new__(Prompt2GuardMethod)
    method.enable_prev_prompt = True
    method.task_ids = []
    method.current_task_id = None
    method.current_task_index = -1
    method.network = _TinyPromptNetwork()

    method.before_task(SimpleNamespace(task_id=0))
    method.network.prompt_learner[0].weight.data.fill_(42.0)
    method.before_task(SimpleNamespace(task_id=1))

    assert torch.equal(method.network.prompt_learner[1].weight, torch.tensor([42.0]))
    assert method.current_task_index == 1
    assert method.network.frozen_indices == [0, 1]


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
