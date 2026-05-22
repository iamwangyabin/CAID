from __future__ import annotations

import torch
import torch.nn.functional as F


def _rbf_kernel(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    x = x.reshape(x.shape[0], -1).float()
    sq = torch.cdist(x, x, p=2).pow(2)
    if sigma is None:
        with torch.no_grad():
            vals = sq.detach().flatten()
            vals = vals[vals > 0]
            sigma = torch.sqrt(torch.median(vals)).item() if vals.numel() else 1.0
            sigma = max(float(sigma), 1e-6)
    return torch.exp(-sq / (2.0 * sigma * sigma))


def _linear_kernel(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(x.shape[0], -1).float()
    x = x - x.mean(dim=0, keepdim=True)
    return x @ x.T


def hsic(x: torch.Tensor, y: torch.Tensor, x_kernel: str = "rbf", y_kernel: str = "linear", unbiased: bool = False) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion.

    The biased estimator is stable for minibatch training and is used by default.
    """
    n = x.shape[0]
    if n < 2:
        return x.new_tensor(0.0)
    k = _rbf_kernel(x) if x_kernel == "rbf" else _linear_kernel(x)
    l = _rbf_kernel(y.float()) if y_kernel == "rbf" else _linear_kernel(y.float())
    if unbiased and n > 3:
        k = k.clone(); l = l.clone()
        k.fill_diagonal_(0); l.fill_diagonal_(0)
        term1 = (k * l).sum()
        term2 = k.sum() * l.sum() / ((n - 1) * (n - 2))
        term3 = 2 * (k.sum(dim=0) * l.sum(dim=0)).sum() / (n - 2)
        return (term1 + term2 - term3) / (n * (n - 3))
    h = torch.eye(n, device=x.device, dtype=x.dtype) - torch.ones(n, n, device=x.device, dtype=x.dtype) / n
    return torch.trace(k @ h @ l @ h) / ((n - 1) ** 2)


def hsic_bottleneck_loss(
    features: torch.Tensor,
    labels_onehot: torch.Tensor | None = None,
    nuisances: list[torch.Tensor] | None = None,
    lambda_label: float = 0.0,
    lambda_nuisance: float = 1.0,
) -> torch.Tensor:
    """HSIC bottleneck objective.

    CE normally provides label supervision.  If `lambda_label > 0`, dependence
    with labels is rewarded by subtracting HSIC(features, labels). Dependence
    with nuisances such as generator ID or caption alignment is penalized.
    """
    loss = features.new_tensor(0.0)
    if labels_onehot is not None and lambda_label:
        loss = loss - lambda_label * hsic(features, labels_onehot, y_kernel="linear")
    if nuisances:
        for n in nuisances:
            if n is None:
                continue
            if n.dim() == 1:
                n = F.one_hot(n.long(), num_classes=int(n.max().item()) + 1).float()
            loss = loss + lambda_nuisance * hsic(features, n.float(), y_kernel="linear")
    return loss
