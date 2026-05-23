from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from caidbench.config import load_config
from caidbench.methods.hdp import HDPMethod


def _image_loader() -> DataLoader:
    x = torch.zeros(4, 3, 8, 8, dtype=torch.float32)
    x[0].fill_(-0.8)
    x[1].fill_(0.8)
    x[2].fill_(-0.6)
    x[3].fill_(0.6)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return DataLoader([{"x": x[i], "y": y[i]} for i in range(x.shape[0])], batch_size=2, shuffle=False)


def _method() -> HDPMethod:
    return HDPMethod(
        epsilon=0.05,
        uap_shape=[3, 8, 8],
        uap_iters=1,
        uap_success_threshold=0.0,
        uap_max_steps_per_batch=1,
        clamp_inputs=False,
        detector_cfg={"num_classes": 2, "backbone": {"type": "small_conv", "out_dim": 4}},
    )


def test_hdp_generates_persistent_uap_pool_and_replays_pseudo_features():
    method = _method()
    loader = _image_loader()

    method.before_task(0, loader)
    method.after_task(0, loader)

    assert method.uap_count == 1
    assert method.uap_pool.shape == (1, 1, 3, 8, 8)
    assert method.teacher is not None

    batch = next(iter(loader))
    out = method.observe(batch, task=1)

    assert torch.isfinite(out["loss"])
    assert out["pseudo_count"].item() == 1.0
    assert "real_feature_kd" in out
    assert "pseudo_feature_kd" in out

    restored = _method()
    restored.load_state_dict(method.state_dict(), strict=False)
    assert restored.uap_count == 1
    assert torch.equal(restored.uap_pool, method.uap_pool)


def test_hdp_default_config_matches_official_backbone_and_uap_hyperparams():
    cfg = load_config("configs/hdp.yaml")

    assert cfg["scenario"]["transform"]["trsf"][0]["size"] == 224
    assert cfg["scenario"]["transform"]["trsf"][2]["mean"] == [0.5, 0.5, 0.5]
    assert cfg["train"]["optimizer"]["type"] == "adam"
    assert cfg["train"]["optimizer"]["lr"] == 0.0002
    assert cfg["train"]["lr_scheduler"] == "step"
    assert cfg["method"]["epsilon"] == 0.15
    assert cfg["method"]["uap_alpha"] == 0.0001
    assert cfg["method"]["uap_success_threshold"] == 0.8
    assert cfg["method"]["uap_shape"] == [3, 224, 224]
    assert cfg["method"]["detector_cfg"]["backbone"]["name"] == "tf_efficientnet_b4_ns"
    assert cfg["method"]["detector_cfg"]["backbone"]["out_dim"] == 128
    assert cfg["method"]["detector_cfg"]["backbone"]["drop_rate"] == 0.2
