from __future__ import annotations

import pytest
import torch

from caidbench.engine.trainer import Trainer


def test_batch_accuracy_metric_uses_current_batch() -> None:
    output = {"logits": torch.tensor([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])}
    batch = {"y": torch.tensor([1, 1, 1])}

    metrics = Trainer._batch_accuracy_metric(output, batch)

    assert metrics["acc"] == pytest.approx(2 / 3)
