from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07, eps: float = 1e-8) -> torch.Tensor:
    """Supervised contrastive loss for domain/class-invariant representation learning.

    Accepts `features` as [B, D] or [B, V, D]. For [B, V, D], views are flattened
    and labels are repeated.
    """
    if features.dim() == 3:
        b, v, d = features.shape
        features = features.reshape(b * v, d)
        labels = labels.view(-1, 1).repeat(1, v).reshape(-1)
    labels = labels.view(-1)
    features = F.normalize(features, dim=-1)
    sim = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(sim, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    label_mask = labels[:, None].eq(labels[None, :]) & logits_mask
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * logits_mask.float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(eps))
    denom = label_mask.float().sum(dim=1).clamp_min(1.0)
    loss = -(label_mask.float() * log_prob).sum(dim=1) / denom
    valid = label_mask.any(dim=1)
    if valid.any():
        return loss[valid].mean()
    return features.new_tensor(0.0)
