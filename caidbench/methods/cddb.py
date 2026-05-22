from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..memory import ReplayBuffer
from ..registry import register_method
from .base import ContinualMethod, batch_to_device, merge_batches


def cddb_binary_loss(logits: torch.Tensor, labels: torch.Tensor, mode: str = "ce") -> torch.Tensor:
    """Binary losses exposed by the CDDB code path.

    CDDB reports several binary formulations. This helper keeps them explicit:
      * ce / bce: standard two-class CE.
      * sum_a_sig: sum real/fake logits before sigmoid-style BCE.
      * sum_b_sig: feature/logit sum with sigmoid BCE.
      * sum_b_log: log-sum-exp binary criterion.
      * max: max real/fake logit aggregation.
    The variants are intentionally implemented over logits so they work for
    DyTox/LUCIR/iCaRL-style heads and simple binary detectors alike.
    """
    mode = mode.lower()
    y = labels.float().view(-1)
    if logits.ndim == 1 or logits.shape[1] == 1:
        s = logits.view(-1)
        return F.binary_cross_entropy_with_logits(s, y)
    if mode in {"ce", "cross_entropy", "softmax"}:
        return F.cross_entropy(logits, labels.long())
    real_score = logits[:, 0]
    fake_score = logits[:, 1:].sum(dim=1) if logits.shape[1] > 2 else logits[:, 1]
    if mode in {"sum_a_sig", "sum_b_sig"}:
        return F.binary_cross_entropy_with_logits(fake_score - real_score, y)
    if mode in {"sum_b_log", "sum_log"}:
        score = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
        return F.binary_cross_entropy_with_logits(score, y)
    if mode == "max":
        score = logits[:, 1:].max(dim=1).values - logits[:, 0]
        return F.binary_cross_entropy_with_logits(score, y)
    raise KeyError(f"Unknown CDDB binary loss mode: {mode}")


@register_method("cddb")
@register_method("cddb_baseline")
class CDDBBenchmarkMethod(ContinualMethod):
    """CDDB-compatible continual deepfake benchmark baseline.

    This module supplies the CDDB training/evaluation conventions: ordered task
    scenarios, binary deepfake loss variants, and fixed-size exemplar rehearsal.
    Model-specific extensions such as DyTox/LUCIR/iCaRL can be plugged in by
    replacing the detector/head while retaining the same task protocol.
    """

    def __init__(self, binary_loss: str = "ce", memory_size: int = 0, memory_batch_size: int = 32, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.binary_loss = str(binary_loss)
        self.memory = ReplayBuffer(memory_size, balanced=True, group_key="label")
        self.memory_batch_size = int(memory_batch_size)

    def observe(self, batch: dict[str, Any], task: Any | None = None) -> dict[str, torch.Tensor]:
        batch = batch_to_device(batch, self.device)
        mem = self.memory.sample(self.memory_batch_size, device=self.device)
        train_batch = merge_batches(batch, mem)
        out = self.predict(train_batch)
        loss = cddb_binary_loss(out["logits"], train_batch["y"], self.binary_loss)
        return {"loss": loss, "ce": loss.detach()}

    def after_task(self, task: Any, train_loader: Any | None = None) -> None:
        if train_loader is not None and self.memory.capacity > 0:
            for batch in train_loader:
                self.memory.add_batch(batch)
