from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from caidbench.methods.ca_adapter_cail import ContentAgnosticAdapterCAIL
from caidbench.methods.hsic_bottleneck import HSICBottleneckMethod
from caidbench.methods.layup import LayUPMethod, RidgeAccumulator
from caidbench.methods.sur_lid import SURLIDMethod, _kd_loss


def _detector_cfg(out_dim: int = 4) -> dict:
    return {"num_classes": 2, "backbone": {"type": "small_conv", "out_dim": out_dim}}


def test_hsic_nuisance_ids_accept_tensor_metadata_and_task_fallback():
    method = HSICBottleneckMethod(detector_cfg=_detector_cfg())
    device = torch.device("cpu")

    ids = method._nuisance_ids({"generator": torch.tensor([2, 3]), "y": torch.tensor([0, 1])}, device)
    assert ids.tolist() == [2, 3]

    method.current_task_id = 5
    fallback = method._nuisance_ids({"x": torch.zeros(2, 3, 8, 8), "y": torch.tensor([0, 1])}, device)
    assert fallback.tolist() == [5, 5]


def test_ca_adapter_domain_ids_accept_tensor_metadata():
    method = ContentAgnosticAdapterCAIL(detector_cfg=_detector_cfg(), memory_size=0)
    ids = method._domain_ids({"generator": torch.tensor([4, 7])}, torch.device("cpu"))
    assert ids is not None
    assert ids.tolist() == [4, 7]


def test_sur_lid_kd_uses_log_target_kl_divergence():
    student = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], dtype=torch.float32)
    teacher = torch.tensor([[1.5, -0.25], [0.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    temperature = 4.0
    alpha = 0.25

    loss = _kd_loss(student, labels, teacher, temperature=temperature, alpha=alpha)
    expected = nn.KLDivLoss(reduction="batchmean", log_target=True)(
        F.log_softmax(student / temperature, dim=1),
        F.log_softmax(teacher / temperature, dim=1),
    ) * (temperature * temperature * 2.0 * alpha) + F.cross_entropy(student, labels) * (1.0 - alpha)
    assert torch.allclose(loss, expected)


def test_sur_lid_center_replay_selects_requested_count():
    method = SURLIDMethod(detector_cfg=_detector_cfg(), max_tasks=1, replay_mode="center")
    z = torch.eye(4)
    center = torch.ones(4)
    idx = method._select_sparse_uniform(z, center, k=1)
    assert idx.numel() == 1


def test_layup_ridge_selection_uses_stratified_accuracy_cv():
    ridge = RidgeAccumulator(feature_dim=2, num_classes=2)
    x = torch.tensor(
        [
            [2.0, 0.1],
            [1.8, 0.2],
            [2.2, -0.1],
            [1.9, 0.0],
            [0.1, 2.0],
            [0.2, 1.8],
            [-0.1, 2.2],
            [0.0, 1.9],
        ]
    )
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    selected = ridge.select_ridge_stratified_accuracy(x, y, [1e-8, 1e8], n_splits=4)
    assert selected == 1e-8


def test_layup_ridge_loader_uses_test_transform_without_shuffle():
    calls = []

    class Source:
        def make_dataset(self, indices, transform_cfg=None, task_id=None, task_name=None):
            calls.append(
                {
                    "indices": list(indices),
                    "transform_cfg": transform_cfg,
                    "task_id": task_id,
                    "task_name": task_name,
                }
            )
            return [{"x": torch.zeros(3, 8, 8), "y": 0}]

    task = SimpleNamespace(task_id=3, name="GauGAN")
    scenario = SimpleNamespace(
        source=Source(),
        tasks=[task],
        _split_indices={(0, "train"): [5, 2, 9]},
        _transform_for_split=lambda split: f"{split}_transform",
    )
    trainer = SimpleNamespace(scenario=scenario, batch_size=7, num_workers=0)
    method = object.__new__(LayUPMethod)
    method.use_test_transform_for_ridge = True

    loader = method._build_ridge_loader(trainer, task, fallback_loader=None)

    assert calls == [
        {
            "indices": [5, 2, 9],
            "transform_cfg": "test_transform",
            "task_id": 3,
            "task_name": "GauGAN",
        }
    ]
    assert loader.batch_size == 7
    assert loader.sampler.__class__.__name__ == "SequentialSampler"
