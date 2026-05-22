from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0, reduction: str = "batchmean") -> torch.Tensor:
    """KL distillation loss with the standard T^2 scaling."""
    t = float(temperature)
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    t_prob = F.softmax(teacher_logits.detach() / t, dim=-1)
    return F.kl_div(s_logp, t_prob, reduction=reduction) * (t * t)


def feature_distillation_loss(student_features: torch.Tensor, teacher_features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    if normalize:
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features.detach(), dim=-1)
    return F.mse_loss(student_features, teacher_features)
