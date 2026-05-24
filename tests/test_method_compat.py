from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from caidbench.methods.ca_adapter_cail import ContentAgnosticAdapterCAIL
from caidbench.methods.hsic_bottleneck import HSICBottleneckMethod
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
