from __future__ import annotations

import pytest
import torch

from caidbench.engine.trainer import Trainer


def _trainer(lr_scheduler: str) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.lr_scheduler_cfg = lr_scheduler
    trainer.max_epochs = 2
    return trainer


def test_cosine_scheduler_decays_lr_by_step() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = _trainer("cosine").make_scheduler(optimizer, total_steps=4)

    assert scheduler is not None

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0853553391)

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_none_scheduler_is_disabled() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.1)

    assert _trainer("none").make_scheduler(optimizer) is None
