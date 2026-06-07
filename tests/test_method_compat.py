from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from caidbench.methods.ca_adapter_cail import ContentAgnosticAdapterCAIL
from caidbench.methods.cp_prompt import CPPromptMethod
from caidbench.methods.dce import CosineLinear, DCEMethod, DCESelector
from caidbench.methods.duct import DUCTMethod, _sinkhorn_uniform
from caidbench.methods.hsic_bottleneck import HSICBottleneckMethod
from caidbench.methods.layup import LayUPMethod, RidgeAccumulator
from caidbench.methods.loranpac import LoRanPACMethod, OnlineTruncatedSVDSolver
from caidbench.methods.pina import PINAMethod
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


def test_hsic_official_mode_uses_online_bottleneck_features():
    method = HSICBottleneckMethod(
        detector_cfg=_detector_cfg(out_dim=6),
        objective="official",
        bottleneck_dim=3,
        memory_size=0,
        lambda_x=1.0,
        lambda_y=1.0,
    )
    batch = {"x": torch.randn(4, 3, 16, 16), "y": torch.tensor([0, 1, 0, 1])}

    out = method.predict(batch)
    log = method.observe(batch)

    assert out["input_features"].shape == (4, 6)
    assert out["features"].shape == (4, 3)
    assert out["logits"].shape == (4, 1)
    assert torch.isfinite(log["loss"])


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


def test_loranpac_rank_schedule_uses_total_samples_after_official_flush():
    solver = OnlineTruncatedSVDSolver(feature_dim=20, num_classes=2, rank=100, truncate_percent=25)
    solver.update(torch.randn(10, 20), torch.arange(10) % 2)
    assert solver.s.numel() == 0
    solver.finalize()
    assert solver.s.numel() == 8
    solver.update(torch.randn(10, 20), torch.arange(10) % 2)
    solver.finalize()
    assert solver.s.numel() == 15


def test_loranpac_tsvd_loader_uses_test_transform_with_shuffle():
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
    trainer = SimpleNamespace(scenario=scenario, batch_size=2, num_workers=0)
    method = object.__new__(LoRanPACMethod)
    method.use_test_transform_for_tsvd = True
    method.tsvd_batch_size = 7

    loader = method._build_tsvd_loader(trainer, task, fallback_loader=None)

    assert calls == [
        {
            "indices": [5, 2, 9],
            "transform_cfg": "test_transform",
            "task_id": 3,
            "task_name": "GauGAN",
        }
    ]
    assert loader.batch_size == 7
    assert loader.sampler.__class__.__name__ == "RandomSampler"


def test_pina_routes_with_official_l1_distance():
    method = PINAMethod(detector_cfg=_detector_cfg(out_dim=2), num_centers=1)
    method.adapters["task0"] = nn.Identity()
    method.adapters["task1"] = nn.Identity()
    method.register_buffer(method._center_name("task0"), torch.tensor([[0.0, 0.0]]))
    method.register_buffer(method._center_name("task1"), torch.tensor([[0.9, 1.1]]))

    selection = method._route(torch.tensor([[0.0, 1.0]]))

    assert selection.tolist() == [0]


def test_cp_prompt_snapshots_fixed_common_prompt_after_task():
    method = CPPromptMethod(detector_cfg=_detector_cfg(out_dim=4), is_fix_share_prompt=True)
    task = SimpleNamespace(task_id=0, name="task0")
    method.before_task(task)
    with torch.no_grad():
        method.common_prompt.fill_(1.0)
    method.after_task(task, train_loader=None)
    with torch.no_grad():
        method.common_prompt.fill_(2.0)

    snapshot = method._common_prompt_for("task0", torch.zeros(1, 4))

    assert torch.allclose(snapshot, torch.ones(4))


def test_dce_uses_sigma_cosine_head_and_mlp_selector():
    head = CosineLinear(4, 2)
    selector_method = DCEMethod(detector_cfg=_detector_cfg(out_dim=4), total_sessions=2)

    assert head.sigma is not None
    assert isinstance(selector_method.selector, DCESelector)
    assert selector_method.feature_scaling_mode == 1


def test_dce_uses_domain_level_class_counts():
    method = DCEMethod(detector_cfg=_detector_cfg(out_dim=4), total_sessions=2)
    task = SimpleNamespace(task_id=0, name="task0")
    loader = SimpleNamespace(dataset=SimpleNamespace(labels=torch.tensor([0, 0, 0, 0])))

    method.before_task(task, loader)

    assert torch.allclose(method.current_class_counts, torch.tensor([4.0, 0.1]))


def test_dce_reinitializes_selector_like_official_update_fc():
    method = DCEMethod(detector_cfg=_detector_cfg(out_dim=4), total_sessions=4)
    method.before_task(SimpleNamespace(task_id=0, name="task0"), train_loader=None)
    assert method.selector.net[-1].out_features == 3
    first_selector = method.selector

    method.before_task(SimpleNamespace(task_id=1, name="task1"), train_loader=None)

    assert method.selector is not first_selector
    assert method.selector.net[-1].out_features == 6


def test_dce_covariance_matches_official_task_average():
    method = DCEMethod(
        detector_cfg=_detector_cfg(out_dim=4),
        total_sessions=2,
        margin_sample_num=2,
        covariance_jitter=0.0,
    )
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1])

    method._update_stats("task0", features, labels)

    cov0 = method.stats[("task0", 0)][1]
    cov1 = method.stats[("task0", 1)][1]
    assert torch.allclose(cov0, cov1)


def test_duct_sinkhorn_transport_is_doubly_stochastic():
    cost = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    transport = _sinkhorn_uniform(cost, reg=0.1)

    assert torch.allclose(transport.sum(dim=0), torch.full((2,), 0.5, dtype=torch.double), atol=1e-4)
    assert torch.allclose(transport.sum(dim=1), torch.full((2,), 0.5, dtype=torch.double), atol=1e-4)


def test_duct_update_fc_matches_official_dynamic_cosine_head():
    method = DUCTMethod(detector_cfg=_detector_cfg(out_dim=4), increment=2, total_sessions=2)
    method._update_fc(2)
    assert method.expanded_head is not None
    old_weight = torch.full_like(method.expanded_head.weight, 0.25)
    method.expanded_head.weight.data.copy_(old_weight)
    method.expanded_head.sigma.data.fill_(7.0)

    method._update_fc(4)

    assert method.expanded_head.out_features == 4
    assert torch.allclose(method.expanded_head.weight[:2], old_weight)
    assert torch.allclose(method.expanded_head.sigma, torch.ones_like(method.expanded_head.sigma))


def test_duct_transport_classifier_handles_float_head_with_double_sinkhorn():
    method = DUCTMethod(detector_cfg=_detector_cfg(out_dim=4), increment=2, total_sessions=2)
    method._current_index = 1
    method._class_means = {
        0: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 1.0, 0.0, 0.0]),
        2: torch.tensor([0.9, 0.1, 0.0, 0.0]),
        3: torch.tensor([0.1, 0.9, 0.0, 0.0]),
    }

    method._transport_classifier()

    assert method.expanded_head.weight.dtype == torch.float32
    assert torch.isfinite(method.expanded_head.weight[:2]).all()
