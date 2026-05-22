from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_distance_mse(student_features: torch.Tensor, teacher_features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Structure-to-structure KD by matching pairwise feature distance matrices."""
    if student_features.shape[0] < 2:
        return student_features.new_tensor(0.0)
    if normalize:
        student_features = F.normalize(student_features, dim=-1)
        teacher_features = F.normalize(teacher_features.detach(), dim=-1)
    ds = torch.cdist(student_features, student_features, p=2)
    dt = torch.cdist(teacher_features, teacher_features, p=2)
    return F.mse_loss(ds, dt)


def category_alignment_loss(features: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor | None = None, real_label: int = 0, fake_label: int = 1, real_weight: float = 1.0, fake_weight: float = 0.25) -> torch.Tensor:
    """Asymmetric category-aware alignment.

    Real images are encouraged to be compact. Fake images are regularized more
    weakly because fake distributions can be diverse across generators.
    """
    labels = labels.view(-1)
    features = F.normalize(features, dim=-1)
    loss = features.new_tensor(0.0)
    count = 0
    for cls, weight in [(real_label, real_weight), (fake_label, fake_weight)]:
        mask = labels == cls
        if mask.sum() > 1:
            z = features[mask]
            centroid = z.mean(dim=0, keepdim=True)
            loss = loss + weight * (z - centroid).pow(2).sum(dim=1).mean()
            count += 1
    if domains is not None:
        # Align real domains more strongly than fake domains.
        for cls, weight in [(real_label, real_weight), (fake_label, fake_weight)]:
            cls_mask = labels == cls
            if cls_mask.sum() < 2:
                continue
            global_centroid = features[cls_mask].mean(dim=0, keepdim=True)
            for d in torch.unique(domains[cls_mask]):
                m = cls_mask & domains.eq(d)
                if m.sum() > 1:
                    loss = loss + 0.5 * weight * F.mse_loss(features[m].mean(dim=0, keepdim=True), global_centroid.detach())
                    count += 1
    return loss / max(count, 1)
